

#include <nvimage/Image.h>
#include <nvtt/nvtt.h>
#include <cstring>

//#define YUV


inline void bgr_to_coycg(const float rgbin[], float yog[3])
{
    float r = rgbin[2];
    float g = rgbin[1];
    float b = rgbin[0];
    float Y = (r + 2 * g + b) * 0.25f;
    float Co = (2 * r - 2 * b) * 0.25f;
    float Cg = (-r + 2 * g - b) * 0.25f;
    Co *= 2;
    Cg *= 2;
    const float bias = 15.0f / 31;// 127.f / 255;
    yog[2] = bias * (2*Co + 1);
    yog[1] = Y;
    yog[0] = bias * (2*Cg + 1);
}

////////////////////////////////////////////////////////////////////////////////
struct HighPass
{
    bool decompose(const uint8_t* rgbx, uint len, uint pitch, bool srgbin, bool tonorm);
    void reconstruct(int unfiltered);

    void get_image_mips(nvtt::InputOptions* input, bool tosrgb, bool tonorm, bool toyuv);

protected:

    void decompose_rows(const float* rgb1, const float* rgb2, uint len, float* sums, float* diff, uint8_t* gray);
    void compose_rows(float* rgb1, float* rgb2, uint len, const float* sums, const float* diff, float limdif);

    void reconstruct_level(int level, int unfiltered);

    size_t _count = 0;
    float* _sums = 0;
    uint8_t* _wrkgray = 0;
    float* _wavbuf = 0;
    float* _reconst = 0;

    struct pass_info {
        float median[3] = {0,};
        float sqdiff[3] = {0,};
    };

    pass_info _info[32];
    pass_info* _current = 0;

    uint _nb;
    uint _width;
    int _levels;
};


////////////////////////////////////////////////////////////////////////////////
template<int NB, bool SRGB, bool NORM>
static void load_row(const uint8_t* rgbin, float* rgbout, uint len)
{
    len *= 4;
    const float C = 1.0f / 255;

    for (uint i = 0; i < len; i += 4)
    {
        if (NORM) {
            const float CH = 1.0f / 127;
            rgbout[i + 0] = float(*rgbin++ - 127) * CH;
            rgbout[i + 1] = float(*rgbin++ - 127) * CH;
            rgbout[i + 2] = float(*rgbin++ - 127) * CH;
        }
        else {
            float a = float(*rgbin++);
            float b = float(*rgbin++);
            float c = float(*rgbin++);

#if 0
            //round to BC1 resolution
            a = roundf(a * (31 / 255.0f)) * (1.0f / 31);
            b = roundf(b * (63 / 255.0f)) * (1.0f / 63);
            c = roundf(c * (31 / 255.0f)) * (1.0f / 31);
#else
            a *= 1 / 255.0f;
            b *= 1 / 255.0f;
            c *= 1 / 255.0f;
#endif

            if (SRGB) {
                //gamma correction
                rgbout[i + 0] = powf(a, 2.2f);
                rgbout[i + 1] = powf(b, 2.2f);
                rgbout[i + 2] = powf(c, 2.2f);
            }
            else {
                rgbout[i + 0] = a;
                rgbout[i + 1] = b;
                rgbout[i + 2] = c;
            }
        }

        if (NB < 4)
            rgbout[i + 3] = 1.0f;
        else
            rgbout[i + 3] = float(*rgbin++) * C;
    }
}

////////////////////////////////////////////////////////////////////////////////
void HighPass::decompose_rows(const float* rgb1, const float* rgb2, uint len, float* sums, float* diff, uint8_t* gray)
{
    len *= 4;

    for (uint i = 0; i < len; i += 4)
    {
        for (uint k = 0; k < 4; ++k, ++i)
        {
            float a = rgb1[i];
            float b = rgb1[i + 4];
            float c = rgb2[i];
            float d = rgb2[i + 4];

            float sa = (a + b) / 2;
            float db = (a - b);
            float sc = (c + d) / 2;
            float dd = (c - d);

            float sac = (sa + sc) / 2;
            float dac = (sa - sc);
            float sbd = (db + dd) / 2;
            float dbd = (db - dd);

            *sums++ = sac;
            *diff++ = dac;
            *diff++ = sbd;
            *diff++ = dbd;

            float v = fabsf(dac) + fabsf(sbd) + fabsf(dbd);
            *gray++ = uint8_t(v * 255 + 0.5f);
            _current->median[k] += v;
            _current->sqdiff[k] += v * v;
        }

        gray[-1] = 255;
    }
}

////////////////////////////////////////////////////////////////////////////////
static float abs_max(float a, float b, float c)
{
    a = fabsf(a);  b = fabsf(b);  c = fabsf(c);
    float t = a > b ? a : b;
    return t > c ? t : c;
}

static void limit(float& h, float& v, float& c, float atop)
{
    if (atop <= 0) {
        h = v = c = 0;
        return;
    }

    float amax = abs_max(h, v, c);
    if (amax > atop) {
        atop /= amax;
        h *= atop;
        v *= atop;
        c *= atop;
    }
}

static float saturate(float v) { return v < 0 ? 0.f : (v > 1 ? 1.f : v); }


void HighPass::compose_rows(float* rgb1, float* rgb2, uint len, const float* sums, const float* diff, float cf)
{
    len *= 4;

    for (uint i = 0; i < len; i += 4)
    {
        const float* pd = diff;//d;

        for (uint k = 0; k < 4; ++k, ++i)
        {
            float sac = *sums++;
            float dac = cf * *pd++;
            float sbd = cf * *pd++;
            float dbd = cf * *pd++;

            float sa = sac + dac / 2;
            float sc = sac - dac / 2;
            float db = sbd + dbd / 2;
            float dd = sbd - dbd / 2;

            float a = sa + db / 2;
            float b = sa - db / 2;
            float c = sc + dd / 2;
            float d = sc - dd / 2;

            rgb1[i] = a;
            rgb1[i + 4] = b;
            rgb2[i] = c;
            rgb2[i + 4] = d;
        }

        diff += 3 * 4;
    }
}

inline uint int_low_pow2(uint x) {
    return x < 2
        ? 0
        : 1 + int_low_pow2(x >> 1);
}

////////////////////////////////////////////////////////////////////////////////
bool HighPass::decompose(const uint8_t* rgbx, uint len, uint pitch, bool srgbin, bool tonormal)
{
    if ((len & (len - 1)) != 0)
        return false;           //not a power of two

    _levels = int_low_pow2(len);
    _width = len;
    _nb = 4;

    if (pitch == 0)
        pitch = _nb * len;

    _count = (4 * len * len * 4) / 3;

    _sums = (float*)malloc(_count * sizeof(float));
    float* ps = _sums;


    void (*load_row_fn)(const uint8_t*, float*, uint) =
        tonormal ? &load_row<4, false, true> :
        srgbin ? &load_row<4, true, false> : &load_row<4, false, false>;

    float smin[4] = {1, 1, 1, 1};
    float smax[4] = {0, 0, 0, 0};

    for (uint i = 0; i < len; ++i, ps += 4 * len, rgbx += pitch) {
        load_row_fn(rgbx, ps, len);

        const float* pr = ps;
        for (uint j = 0; j < len; ++j, pr += 4) {
            for (int c = 0; c < 4; ++c) {
                if (pr[c] > smax[c]) smax[c] = pr[c];
                if (pr[c] < smin[c]) smin[c] = pr[c];
            }
        }
    }

    pitch = 4 * len;

    _wrkgray = (uint8_t*)malloc((4 * len * len) / 4);
    _wavbuf = (float*)malloc((4 * len * len - 4) * sizeof(float));

    float* pw = _wavbuf;
    const float* pin = _sums;

    for (uint i = _levels + 1; i > 1; )
    {
        --i;
        uint w = 1 << i;

        pass_info& pi = _info[i];
        _current = &pi;
        const float* pso = ps;
        uint8_t* pg = _wrkgray;

        for (uint j = 0; j < w; j += 2) {
            decompose_rows(pin, pin + pitch, w, ps, pw, pg);
            pin += 2 * pitch;
            ps += 4 * (w / 2);
            pg += 4 * (w / 2);
            pw += 3 * 4 * (w / 2);
        }

        float d = 1.0f / (3 * w * w / 4);
        pi.median[0] = d * pi.median[0];
        pi.median[1] = d * pi.median[1];
        pi.median[2] = d * pi.median[2];
        pi.sqdiff[0] = sqrtf(d * pi.sqdiff[0]);
        pi.sqdiff[1] = sqrtf(d * pi.sqdiff[1]);
        pi.sqdiff[2] = sqrtf(d * pi.sqdiff[2]);

        pin = pso;
        pitch = 4 * w / 2;
    }

    //_sums.resize(ps - _sums.ptr());

    ps -= 4;

    if (tonormal) {
        //normal maps: set avg, clear any bias
        ps[0] = 1;
        ps[1] = 0;
        ps[2] = 0;
    }
    else {
        //align the toplevel sum
        uint r = uint(ps[0] * 255 + 0.5);
        uint g = uint(ps[1] * 255 + 0.5);
        uint b = uint(ps[2] * 255 + 0.5);
        ps[0] = r / 255.f;
        ps[1] = g / 255.f;
        ps[2] = b / 255.f;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
void HighPass::reconstruct(int unfiltered)
{
    _reconst = (float*)malloc((_count - 1) * sizeof(float));

    for (int i = 0; i <= _levels; ++i)
    {
        reconstruct_level(i, unfiltered);
    }
}

////////////////////////////////////////////////////////////////////////////////
void HighPass::reconstruct_level(int level, int unfiltered)
{
    const float* ps = _sums + _count - 1;
    const float* pd = _wavbuf + (4 * _width * _width - 4);
    
    float* pr = _reconst + (_count - 1 - 4);
    pr[0] = ps[-4 + 0];
    pr[1] = ps[-4 + 1];
    pr[2] = ps[-4 + 2];
    pr[3] = ps[-4 + 3];

    int levsup = _levels - 1 - level - unfiltered;

    for (int i = 0; i < level; ++i)
    {
        //float c = ldexp(1.0f, int(i-_levels));
        //float c = float(int(i + 1 - (_levels-level))) / level;
        //float limx = limstart * c;

        float cf = i < levsup ? ldexpf(1.0f, i - levsup) : 1.0f;

        uint w = 1 << i;
        uint s = w << i;
        ps = pr;
        pd -= 4 * 3 * s;
        pr -= 4 * 4 * s;

        const float* pso = ps;
        const float* pdo = pd;
        float* pro = pr;
        for (uint k = 0; k < w; ++k, pso += 4 * w, pdo += 4 * 3 * w, pro += 2 * 4 * 2 * w)
            compose_rows(pro, pro + 4 * 2 * w, 2 * w, pso, pdo, cf);
    }

    //update sums
    size_t offs = pr - _reconst;
    ::memcpy(_sums + offs, pr, (4 << level << level) * sizeof(float));
}

////////////////////////////////////////////////////////////////////////////////
void HighPass::get_image_mips(nvtt::InputOptions* input, bool tosrgb, bool tonorm, bool toyuv)
{
    const float* ps = _sums;
    uint8_t* pw = (uint8_t*)malloc(_count);

    //mips.alloc(_levels + 1);

    for (uint i = _levels + 1; i > 0; )
    {
        --i;
        uint width = 1 << i;
        uint size = 4 << i << i;

        //mips[_levels - i] = pw;
        const uint8_t* pwl = pw;
        float fvec[3];

        for (uint k = 0; k < size; k += 4)
        {
            const int K = 2047483673;
            const float IRANGE = float(1.0 / 2147483648.0);

            int p = (K * k + 1) * k;
            float noise = p * IRANGE;   //-1..1

            if (tonorm) {
                float blue2 = 1 - (ps[1] * ps[1] + ps[2] * ps[2]);
                float blue = blue2 > 0 ? sqrtf(blue2) : 0.0f;
                fvec[0] = saturate((blue + 1) * (127 / 255.f));
                fvec[1] = saturate((ps[1] + 1) * (127 / 255.f));
                fvec[2] = saturate((ps[2] + 1) * (127 / 255.f));
            }
            else if (tosrgb || toyuv) {
                fvec[0] = powf(saturate(ps[0]), 1 / 2.2f);
                fvec[1] = powf(saturate(ps[1]), 1 / 2.2f);
                fvec[2] = powf(saturate(ps[2]), 1 / 2.2f);

                if (toyuv) {
                    bgr_to_coycg(fvec, fvec);
                    fvec[1] += (0.5f / 63) * noise;
                }
            }
            else {
                fvec[0] = saturate(ps[0]);
                fvec[1] = saturate(ps[1]);
                fvec[2] = saturate(ps[2]);
            }

            *pw++ = uint8_t(0.5f + 255 * fvec[0]);
            *pw++ = uint8_t(0.5f + 255 * fvec[1]);
            *pw++ = uint8_t(0.5f + 255 * fvec[2]);
            *pw++ = 255;
            ps += 4;
        }

        input->setMipmapData(pwl, width, width, 1, 0, _levels - i);
    }
}


bool high_pass(nvtt::InputOptions* input, nv::Image* image, bool linear, bool to_normal, bool to_yuv, int skip_mips)
{
    HighPass hp;
    if (!hp.decompose((const uint8_t*)image->pixels(), image->width, 0, !linear, to_normal))
        return false;

    hp.reconstruct(skip_mips);// srgb ? nmipmaps - 10 : 0);

    hp.get_image_mips(input, !linear, to_normal, to_yuv);

    return true;
}
