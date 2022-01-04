// Copyright NVIDIA Corporation 2007 -- Ignacio Castano <icastano@nvidia.com>
// 
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
// 
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include "cmdline.h"

#include <nvtt/nvtt.h>

#include <nvimage/Image.h>    // @@ It might be a good idea to use FreeImage directly instead of ImageIO.
#include <nvimage/ImageIO.h>
#include <nvimage/FloatImage.h>
#include <nvimage/DirectDrawSurface.h>
#include <nvimage/HoleFilling.h>

#include <nvcore/Ptr.h> // AutoPtr
#include <nvcore/StrLib.h> // Path
#include <nvcore/StdStream.h>
#include <nvcore/FileSystem.h>
#include <nvcore/Timer.h>
#include <nvcore/Utils.h>

#include <nvmath/Color.h>

#include <zstd/zstd.h>
#include <cctype>

struct MyOutputHandler : public nvtt::OutputHandler
{
    MyOutputHandler(const char * name) : total(0), progress(0), percentage(0), stream(new nv::StdOutputStream(name)) {}
    virtual ~MyOutputHandler() { delete stream; }

    virtual void beginImage(int size, int width, int height, int depth, int face, int miplevel)
    {
        // ignore.
    }

    virtual void endImage()
    {
        // Ignore.
    }

    // Output data.
    virtual bool writeData(const void * data, int size)
    {
        nvDebugCheck(stream != NULL);
        if (data && size > 0)
            stream->serialize(const_cast<void *>(data), size);

        progress += size;
        int p = int((100 * progress) / total);
        if (verbose && p != percentage)
        {
            nvCheck(p >= 0);

            percentage = p;
            printf("\r%d%%", percentage);
            fflush(stdout);
        }

        return true;
    }

    void setTotal(int64 t)
    {
        total = t + 128;
    }
    void setDisplayProgress(bool b)
    {
        verbose = b;
    }

    int64 total;
    int64 progress;
    int percentage;
    bool verbose;
    nv::StdOutputStream * stream;
};


struct ZstdOutputHandler : public MyOutputHandler
{
    ZstdOutputHandler(const char* name) : MyOutputHandler(name) {}
    virtual ~ZstdOutputHandler() {
        if (buffer)
            free(buffer);
        if (cstream)
            ZSTD_freeCStream(cstream);
    }

    // Output data.
    virtual bool writeData(const void* src, int size)
    {
        if (!cstream) {
            cstream = ZSTD_createCStream();
            size_t res = ZSTD_initCStream(cstream, 17);
            if (ZSTD_isError(res))
                return false;

            bufsize = ZSTD_CStreamInSize();
            buffer = realloc(buffer, bufsize);
        }

        ZSTD_outBuffer zout;
        zout.pos = offset;
        zout.size = bufsize;
        zout.dst = buffer;

        //if src is null, flush
        if (src == 0) {
            while (ZSTD_endStream(cstream, &zout) > 0) {
                stream->serialize(zout.dst, (uint)zout.pos);
                zout.pos = 0;
            }

            if (zout.pos > 0)
                stream->serialize(zout.dst, (uint)zout.pos);
            offset = 0;

            return true;
        }

        ZSTD_inBuffer zin;
        zin.src = src;
        zin.size = size;
        zin.pos = 0;

        do {
            size_t res = ZSTD_compressStream(cstream, &zout, &zin);
            if (ZSTD_isError(res))
                return false;

            if (zout.pos >= zout.size) {
                stream->serialize(zout.dst, (uint)zout.pos);
                zout.pos = 0;
            }
        }
        while (zin.size > zin.pos);

        offset = zout.pos;

        /*progress += size;
        int p = int((100 * progress) / total);
        if (verbose && p != percentage)
        {
            nvCheck(p >= 0);

            percentage = p;
            printf("\r%d%%", percentage);
            fflush(stdout);
        }*/

        return true;
    }

    ZSTD_CStream* cstream = 0;
    size_t offset = 0;
    size_t bufsize = 0;
    void* buffer = 0;
};


struct MyErrorHandler : public nvtt::ErrorHandler
{
    virtual void error(nvtt::Error e)
    {
#if _DEBUG
        nvDebugBreak();
#endif
        printf("Error: '%s'\n", nvtt::errorString(e));
    }
};


bool high_pass(nvtt::InputOptions* input, nv::Image* image, bool linear, bool to_normal, int to_yuv, int skip_mips);


// Set color to normal map conversion options.
void setColorToNormalMap(nvtt::InputOptions & inputOptions)
{
    inputOptions.setNormalMap(false);
    inputOptions.setConvertToNormalMap(true);
    inputOptions.setHeightEvaluation(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 0.0f);
    //inputOptions.setNormalFilter(1.0f, 0, 0, 0);
    //inputOptions.setNormalFilter(0.0f, 0, 0, 1);
    inputOptions.setGamma(1.0f, 1.0f);
    inputOptions.setNormalizeMipmaps(true);
}

// Set options for normal maps.
void setNormalMap(nvtt::InputOptions & inputOptions)
{
    inputOptions.setNormalMap(true);
    inputOptions.setConvertToNormalMap(false);
    inputOptions.setGamma(1.0f, 1.0f);
    inputOptions.setNormalizeMipmaps(true);
}

// Set options for color maps.
void setColorMap(nvtt::InputOptions & inputOptions)
{
    inputOptions.setNormalMap(false);
    inputOptions.setConvertToNormalMap(false);
    inputOptions.setGamma(2.2f, 2.2f);
    inputOptions.setNormalizeMipmaps(false);
}

// Set options for linear maps.
void setLinearMap(nvtt::InputOptions & inputOptions)
{
	inputOptions.setNormalMap(false);
	inputOptions.setConvertToNormalMap(false);
	inputOptions.setGamma(1.0f, 1.0f);
	inputOptions.setNormalizeMipmaps(false);
}

// convert surface to image
void toNvImage(const nvtt::Surface& from, nv::Image& to)
{
    nv::Color32* data = new nv::Color32[from.width() * from.height()];
    const int w = from.width();
    const int h = from.height();
    const float* r = from.channel(0);
    const float* g = from.channel(1);
    const float* b = from.channel(2);
    const float* a = from.channel(3);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            int idx = i + j * w;
            data[idx].r = uint8(nv::clamp(r[idx], 0.f, 1.f) * 255.f + 0.5);
            data[idx].g = uint8(nv::clamp(g[idx], 0.f, 1.f) * 255.f + 0.5);
            data[idx].b = uint8(nv::clamp(b[idx], 0.f, 1.f) * 255.f + 0.5);
            data[idx].a = uint8(nv::clamp(a[idx], 0.f, 1.f) * 255.f + 0.5);
        }
    }
    to.acquire(data, w, h);
}

int main(int argc, char *argv[])
{
    MyAssertHandler assertHandler;
    MyMessageHandler messageHandler;

    bool alpha = false;
    bool normal = false;
    bool color2normal = false;
    bool linear = false;
    bool wrapRepeat = false;
    bool noMipmaps = false;
    bool fast = false;
    bool nocuda = false;
    bool bc1n = false;
    bool luminance = false;
    nvtt::Format format = nvtt::Format_Unknown;
    bool fillHoles = false;
    bool outProvided = false;
    bool premultiplyAlpha = false;
    bool highPassMips = false;
    bool highPassYUV = false;
    bool highPassYUVNorm = false;
    int highPassSkip = 0;
    float scaleCoverage[4] = {-1, -1, -1, -1};
    bool scaleCoverageChannels[4] = {false, false, false, false};
    nvtt::MipmapFilter mipmapFilter = nvtt::MipmapFilter_Box;
    bool rgbm = false;
    bool rangescale = false;
    bool srgb = false;

    const char * externalCompressor = NULL;

    bool silent = false;
    bool dds10 = false;
    bool ktx = false;
    bool zstd = false;
    bool argerror = false;

    nv::Path input;
    nv::Path output;
    nv::Path input_normal_for_roughness;

    // Parse arguments.
    for (int i = 1; i < argc; i++)
    {
        // Input options.
        if (strcmp("-color", argv[i]) == 0)
        {
        }
        else if (strcmp("-alpha", argv[i]) == 0)
        {
            alpha = true;
        }
        else if (strcmp("-normal", argv[i]) == 0)
        {
            normal = true;
        }
        else if (strcmp("-tonormal", argv[i]) == 0)
        {
            color2normal = true;
        }
		else if (strcmp("-linear", argv[i]) == 0)
		{
			linear = true;
		}
        else if (strcmp("-clamp", argv[i]) == 0)
        {
        }
        else if (strcmp("-repeat", argv[i]) == 0)
        {
            wrapRepeat = true;
        }
        else if (strcmp("-nomips", argv[i]) == 0)
        {
            noMipmaps = true;
        }
        else if (strcmp("-fillholes", argv[i]) == 0)
        {
            fillHoles = true;
        }
        else if (strcmp("-premula", argv[i]) == 0)
        {
            premultiplyAlpha = true;
        }
        else if (strcmp("-normal_to_roughness", argv[i]) == 0)
        {
            if (i+1 == argc) break;
            i++;

            input_normal_for_roughness = argv[i];
        }
        else if (strcmp("-high_pass", argv[i]) == 0)
        {
            highPassMips = true;

            if (i + 1 < argc) {
                const char* arg1 = argv[i + 1];

                //read optional skip count
                if (isdigit(arg1[0]) || ((arg1[0] == '-' || arg1[0] == '+') && isdigit(arg1[1]))) {
                    i++;
                    char* end;
                    int skip = strtol(argv[i], &end, 10);

                    if (*end != 0) {
                        printf("Unrecognized characters: %s\n", end);
                        argerror = true;
                        break;
                    }

                    highPassSkip = skip;
                }
            }
        }
        else if (strcmp("-yuv", argv[i]) == 0) {
            highPassYUV = true;
            highPassYUVNorm = false;
        }
        else if (strcmp("-yuvn", argv[i]) == 0) {
            
            highPassYUV = true;
            highPassYUVNorm = true;
        }
        else if (strcmp("-coverage", argv[i]) == 0)
        {
            for (int k = 0; k<4; ++k) {
                if (i+1 == argc || !isdigit(argv[i+1][0])) break;
                i++;

                char* end;
                float coverage = strtof(argv[i], &end);

                if (*end != 0) {
                    printf("Unrecognized characters: %s\n", end);
                    argerror = true;
                    break;
                }

                if (i+1 == argc) {
                    printf("Expecting channel number after the coverage value\n");
                    argerror = true;
                    break;
                }
                i++;

                unsigned int ch = strtoul(argv[i], &end, 10);
                if (*end != 0 || ch > 3) {
                    printf("Invalid channel number: %s\n", argv[i]);
                    argerror = true;
                    break;
                }

                scaleCoverage[ch] = coverage;
                scaleCoverageChannels[ch] = true;
            }
        }
        else if (strcmp("-mipfilter", argv[i]) == 0)
        {
            if (i+1 == argc) break;
            i++;

            if (strcmp("box", argv[i]) == 0) mipmapFilter = nvtt::MipmapFilter_Box;
            else if (strcmp("triangle", argv[i]) == 0) mipmapFilter = nvtt::MipmapFilter_Triangle;
            else if (strcmp("kaiser", argv[i]) == 0) mipmapFilter = nvtt::MipmapFilter_Kaiser;
            else {
                printf("Unrecognized filter: %s", argv[i]);
                argerror = true;
            }
        }
        else if (strcmp("-rgbm", argv[i]) == 0)
        {
            rgbm = true;
        }
        else if (strcmp("-rangescale", argv[i]) == 0)
        {
            rangescale = true;
        }


        // Compression options.
        else if (strcmp("-fast", argv[i]) == 0)
        {
            fast = true;
        }
        else if (strcmp("-nocuda", argv[i]) == 0)
        {
            nocuda = true;
        }
        else if (strcmp("-rgb", argv[i]) == 0)
        {
            format = nvtt::Format_RGB;
        }
        else if (strcmp("-lumi", argv[i]) == 0)
        {
            luminance = true;
            format = nvtt::Format_RGB;
        }
        else if (strcmp("-bc1", argv[i]) == 0)
        {
            format = nvtt::Format_BC1;
        }
        else if (strcmp("-bc1n", argv[i]) == 0)
        {
            format = nvtt::Format_BC1;
            bc1n = true;
        }
        else if (strcmp("-bc1a", argv[i]) == 0)
        {
            format = nvtt::Format_BC1a;
        }
        else if (strcmp("-bc2", argv[i]) == 0)
        {
            format = nvtt::Format_BC2;
        }
        else if (strcmp("-bc3", argv[i]) == 0)
        {
            format = nvtt::Format_BC3;
        }
        else if (strcmp("-bc3n", argv[i]) == 0)
        {
            format = nvtt::Format_BC3n;
        }
        else if (strcmp("-bc4", argv[i]) == 0)
        {
            format = nvtt::Format_BC4;
        }
        else if (strcmp("-bc5", argv[i]) == 0)
        {
            format = nvtt::Format_BC5;
        }
        else if (strcmp("-bc6", argv[i]) == 0)
        {
            format = nvtt::Format_BC6;
        }
        else if (strcmp("-bc7", argv[i]) == 0)
        {
            format = nvtt::Format_BC7;
        }
        else if (strcmp("-bc3_rgbm", argv[i]) == 0)
        {
            format = nvtt::Format_BC3_RGBM;
            rgbm = true;
        }
        else if (strcmp("-etc1", argv[i]) == 0)
        {
            format = nvtt::Format_ETC1;
        }
        else if (strcmp("-etc2", argv[i]) == 0 || strcmp("-etc2_rgb", argv[i]) == 0)
        {
            format = nvtt::Format_ETC2_RGB;
        }
        else if (strcmp("-etc2_eac", argv[i]) == 0 || strcmp("-etc2_rgba", argv[i]) == 0)
        {
            format = nvtt::Format_ETC2_RGBA;
        }
        else if (strcmp("-eac", argv[i]) == 0 || strcmp("-etc2_r", argv[i]) == 0)
        {
            format = nvtt::Format_ETC2_R;
        }
        else if (strcmp("-etc2_rg", argv[i]) == 0)
        {
            format = nvtt::Format_ETC2_R;
        }
        else if (strcmp("-etc2_rgbm", argv[i]) == 0)
        {
            format = nvtt::Format_ETC2_RGBM;
            rgbm = true;
        }

        // Undocumented option. Mainly used for testing.
        else if (strcmp("-ext", argv[i]) == 0)
        {
            if (i+1 < argc && argv[i+1][0] != '-') {
                externalCompressor = argv[i+1];
                i++;
            }
        }
        else if (strcmp("-pause", argv[i]) == 0)
        {
            printf("Press ENTER\n"); fflush(stdout);
            getchar();
        }

        // Output options
        else if (strcmp("-silent", argv[i]) == 0)
        {
            silent = true;
        }
        else if (strcmp("-dds10", argv[i]) == 0)
        {
            dds10 = true;
        }
        else if (strcmp("-ktx", argv[i]) == 0)
        {
            ktx = true;
        }
        else if (strcmp("-zstd", argv[i]) == 0)
        {
            zstd = true;
        }
        else if (strcmp("-srgb", argv[i]) == 0)
        {
            srgb = true;
        }
        
        else if (argv[i][0] != '-')
        {
            input = argv[i];

            if (i+1 < argc && argv[i+1][0] != '-') {
                output = argv[i+1];
                if (output.endsWith("\\") || output.endsWith("/")) {
                    //only path specified
                    output.append(input.fileName());
				    output.stripExtension();
				    output.append(zstd ? ".zds" : ".dds");
                }
                else {
                    outProvided = true;
                    if (output.endsWith(".zds"))
                        zstd = true;
                }
            }
            else
            {
                output.copy(input.str());
                output.stripExtension();
                
                if (ktx)
                {
                    output.append(".ktx");
                }
                else
                {
                    output.append(zstd ? ".zds" : ".dds");
                }
            }

            break;
        }
		else
		{
			printf("Warning: unrecognized option \"%s\"\n", argv[i]);
            argerror = true;
		}
    }

    if (argerror) {
        printf("Invalid arguments\n");
        return EXIT_FAILURE;
    }

    if (zstd && !output.endsWith(".zds")) {
        output.stripExtension();
        output.append(".zds");
    }

    const uint version = nvtt::version();
    const uint major = version / 100 / 100;
    const uint minor = (version / 100) % 100;
    const uint rev = version % 100;


    if (!silent)
    {
        printf("NVIDIA Texture Tools %u.%u.%u - Copyright NVIDIA Corporation 2007\n\n", major, minor, rev);
    }

    if (input.isNull())
    {
        printf("usage: nvcompress [options] infile [outfile.dds]\n");

        printf("nInput options:\n");
        printf("  -color        The input image is a color map (default).\n");
        printf("  -alpha        The input image has an alpha channel used for transparency.\n");
        printf("  -normal       The input image is a normal map.\n");
        printf("  -linear       The input is in linear color space.\n");
        printf("  -tonormal     Convert input to normal map.\n");
        printf("  -clamp        Clamp wrapping mode (default).\n");
        printf("  -repeat       Repeat wrapping mode.\n");
        printf("  -nomips       Disable mipmap generation.\n");
        printf("  -coverage     coverage value in range <0; 1>, mipmaps will have the same coverage.\n");
        printf("                second parameter is number of channel to use. Multiple pairs of coverage and channel id can be specified.\n");
        printf("  -high_pass    [optional mip offset]; apply high-pass mipmap filtering.\n");
        printf("  -yuv, -yuvn   highpass options: convert to CoYCg, convert to CoYCg normalized to gray.\n");
        printf("  -premula      Premultiply alpha into color channel.\n");
        printf("  -mipfilter    Mipmap filter. One of the following: box, triangle, kaiser.\n");
        printf("  -rgbm         Transform input to RGBM.\n");
        printf("  -rangescale   Scale image to use entire color range.\n");
        printf("  -fillholes    Fill transparent areas with nearby color.\n");
        printf(" infile1+infile2[+infile3] combine multiple channels into one image, taking the first channel from each.");

        printf("\nCompression options:\n");
        printf("  -fast         Fast compression.\n");
        printf("  -nocuda       Do not use cuda compressor.\n");
        printf("  -rgb          RGBA format\n");
        printf("  -lumi         LUMINANCE format\n");
        printf("  -bc1          BC1 format (DXT1)\n");
        printf("  -bc1n         BC1 normal map format (DXT1nm)\n");
        printf("  -bc1a         BC1 format with binary alpha (DXT1a)\n");
        printf("  -bc2          BC2 format (DXT3)\n");
        printf("  -bc3          BC3 format (DXT5)\n");
        printf("  -bc3n         BC3 normal map format (DXT5nm)\n");
        printf("  -bc4          BC4 format (ATI1)\n");
        printf("  -bc5          BC5 format (3Dc/ATI2)\n");
        printf("  -bc6          BC6 format\n");
        printf("  -bc7          BC7 format\n");
        printf("  -bc3_rgbm     BC3-rgbm format\n");

        printf("\nOutput options:\n");
        printf("  -silent  \tDo not output progress messages\n");
        printf("  -dds10   \tUse DirectX 10 DDS format (enabled by default for BC6/7, unless ktx is being used)\n");
        printf("  -ktx     \tUse KTX container format\n");
        printf("  -zstd    \tApply Zstd compression, produces zds files instead of dds\n");
        printf("  -srgb    \tIf the requested format allows it, output will be in sRGB color space\n\n");

        return EXIT_FAILURE;
    }

    // Make sure input file exists.
    const char* multi = strchr(input.str(), '+');

    if (!multi && !nv::FileSystem::exists(input.str()))
    {
        fprintf(stderr, "The file '%s' does not exist.\n", input.str());
        return 1;
    }

    // Set input options.
    nvtt::InputOptions inputOptions;

    bool useSurface = false;    // @@ use Surface API in all cases!
    nvtt::Surface image;

    if (format == nvtt::Format_Unknown && nv::strCaseDiff(input.extension(), ".dds") == 0)
    {
        // Load surface.
        nv::DirectDrawSurface dds;

        if (!dds.load(input.str()))
        {
            fprintf(stderr, "The file '%s' is not a valid DDS file.\n", input.str());
            return EXIT_FAILURE;
        }

        if (!dds.isSupported())
        {
            fprintf(stderr, "The file '%s' is not a supported DDS file.\n", input.str());
            return EXIT_FAILURE;
        }

        //if format not specified, get from dds
        if (dds.isColorsRGB())
            format = nvtt::Format_RGB;
        else if (dds.isColorsLuminance()) {
            luminance = true;
            format = nvtt::Format_RGB;
        }
        else {
            uint cc = dds.header.fourcc;
            switch(cc) {
            case nv::FOURCC_DXT1:   format = nvtt::Format_DXT1; break;
            case nv::FOURCC_DXT3:   format = nvtt::Format_DXT3; break;
            case nv::FOURCC_DXT5:   format = nvtt::Format_DXT5; break;
            case nv::FOURCC_RXGB:   format = nvtt::Format_BC3n; break;
            case nv::FOURCC_ATI1:   format = nvtt::Format_BC4; break;
            case nv::FOURCC_ATI2:   format = nvtt::Format_BC5; break;
            }
        }

        alpha = dds.hasAlpha();
    }


    if (format == nvtt::Format_BC3_RGBM || format == nvtt::Format_ETC2_RGBM || rgbm) {
        useSurface = true;

        if (!image.load(input.str())) {
            fprintf(stderr, "Error opening input file '%s'.\n", input.str());
            return EXIT_FAILURE;
        }

        if (rangescale) {
            // get color range
            float min_color[3], max_color[3];
            image.range(0, &min_color[0], &max_color[0]);
            image.range(1, &min_color[1], &max_color[1]);
            image.range(2, &min_color[2], &max_color[2]);

            //printf("Color range = %.2f %.2f %.2f\n", max_color[0], max_color[1], max_color[2]);

            float color_range = nv::max3(max_color[0], max_color[1], max_color[2]);
            const float max_color_range = 16.0f;

            if (color_range > max_color_range) {
                //Log::print("Clamping color range %f to %f\n", color_range, max_color_range);
                color_range = max_color_range;
            }
            //color_range = max_color_range;  // Use a fixed color range for now.

            for (int i = 0; i < 3; i++) {
                image.scaleBias(i, 1.0f / color_range, 0.0f);
            }
            image.toneMap(nvtt::ToneMapper_Linear, /*parameters=*/NULL); // Clamp without changing the hue.

            // Clamp alpha.
            image.clamp(3);
        }

        if (alpha) {
            image.setAlphaMode(nvtt::AlphaMode_Transparency);
        }

        // To gamma.
        image.toGamma(2);

        if (format != nvtt::Format_BC3_RGBM || format != nvtt::Format_ETC2_RGBM) {
            image.setAlphaMode(nvtt::AlphaMode_None);
            image.toRGBM(1, 0.15f);
        }
    }
    else if (format == nvtt::Format_BC6) {
        //format = nvtt::Format_BC1;
        //fprintf(stderr, "BLABLABLA.\n");
        useSurface = true;

        if (!image.load(input.str())) {
            fprintf(stderr, "Error opening input file '%s'.\n", input.str());
            return EXIT_FAILURE;
        }

        image.setAlphaMode(nvtt::AlphaMode_Transparency);
    }
    else {
        if (nv::strCaseDiff(input.extension(), ".dds") == 0)
        {
            // Load surface.
            nv::DirectDrawSurface dds;
            if (!dds.load(input.str()) || !dds.isValid())
            {
                fprintf(stderr, "The file '%s' is not a valid DDS file.\n", input.str());
                return EXIT_FAILURE;
            }

            if (!dds.isSupported())
            {
                fprintf(stderr, "The file '%s' is not a supported DDS file.\n", input.str());
                return EXIT_FAILURE;
            }

            uint faceCount;
            if (dds.isTexture2D())
            {
                inputOptions.setTextureLayout(nvtt::TextureType_2D, dds.width(), dds.height());
                faceCount = 1;
            }
            else if (dds.isTexture3D())
            {
                inputOptions.setTextureLayout(nvtt::TextureType_3D, dds.width(), dds.height(), dds.depth());
                faceCount = 1;

                nvDebugBreak();
            }
            else if (dds.isTextureCube()) {
                inputOptions.setTextureLayout(nvtt::TextureType_Cube, dds.width(), dds.height());
                faceCount = 6;
            } else {
                nvDebugCheck(dds.isTextureArray());
                inputOptions.setTextureLayout(nvtt::TextureType_Array, dds.width(), dds.height(), 1, dds.arraySize());
                faceCount = dds.arraySize();
                dds10 = ktx ? false : true;
            }

            uint mipmapCount = dds.mipmapCount();

            nv::Image mipmap;

            for (uint f = 0; f < faceCount; f++)
            {
                for (uint m = 0; m < mipmapCount; m++)
                {
                    if (imageFromDDS(&mipmap, dds, f, m)) // @@ Load as float.
                        inputOptions.setMipmapData(mipmap.pixels(), mipmap.width, mipmap.height, mipmap.depth, f, m);
                }
            }
        }
        else
        {
            // Regular image.
            nv::Image image;
            if (!image.load(input.str()))
            {
                fprintf(stderr, "The file '%s' is not a supported image type.\n", input.str());
                return 1;
            }

            if (highPassMips) {
                inputOptions.setTextureLayout(nvtt::TextureType_2D, image.width, image.height);

                int yuv = !highPassYUV ? 0 :
                    highPassYUVNorm ? -1 : 1;
                if (!high_pass(&inputOptions, &image, linear || normal, normal, yuv, highPassSkip)) {
                    fprintf(stderr, "Error applying high pass filter.\n");
                    return 1;
                }
            }
            else if (!input_normal_for_roughness.isNull()) {
                nvtt::Surface fimage;
                fimage.setImage(nvtt::InputFormat_BGRA_8UB, image.width, image.height, 1, image.pixels());

                nvtt::Surface normal;
                if (!normal.load(input_normal_for_roughness.str())) {
                    fprintf(stderr, "The file '%s' is not a supported image type.\n", input_normal_for_roughness.str());
                    return 1;
                }
                inputOptions.setTextureLayout(nvtt::TextureType_2D, image.width, image.height);
                nv::Image img_0;
                toNvImage(fimage, img_0);
                inputOptions.setMipmapData(img_0.pixels(), img_0.width, img_0.height);

                int mip = 1;
                while (fimage.buildNextMipmap(nvtt::MipmapFilter_Box)) {
                    fimage.roughnessMipFromNormal(normal);
                    nv::Image img;
                    toNvImage(fimage, img);
                    inputOptions.setMipmapData(img.pixels(), img.width, img.height, 1, 0, mip);
                    ++mip;
                }
            }
            else if (scaleCoverageChannels[0] || scaleCoverageChannels[1] || scaleCoverageChannels[2] || scaleCoverageChannels[3]) {
                nvtt::Surface fimage;

                fimage.setImage(nvtt::InputFormat_BGRA_8UB, image.width, image.height, 1, image.pixels());
                inputOptions.setTextureLayout(nvtt::TextureType_2D, image.width, image.height);

                nv::Image img_0;
                toNvImage(fimage, img_0);
                inputOptions.setMipmapData(img_0.pixels(), img_0.width, img_0.height);

                float coverage0[4] = {0,};
                for (int k = 0; k < 4; ++k) {
                    if (scaleCoverageChannels[k])
                        coverage0[k] = fimage.alphaTestCoverage(scaleCoverage[k], k);
                }

                int mip = 1;
                while (fimage.buildNextMipmap(nvtt::MipmapFilter_Box)) {
                    nvtt::Surface mip_img;
                    mip_img.setImage(fimage.width(), fimage.height(), 1);
                    mip_img.copy(fimage, 0, 0, 0, fimage.width(), fimage.height(), 1, 0, 0, 0);

                    for (int k = 0; k < 4; ++k) {
                        if (scaleCoverageChannels[k])
                            mip_img.scaleAlphaToCoverage(coverage0[k], scaleCoverage[k], k);
                    }

                    nv::Image img;
                    toNvImage(mip_img, img);
                    inputOptions.setMipmapData(img.pixels(), img.width, img.height, 1, 0, mip);
                    ++mip;
                }
            }
            else if (fillHoles) {
                nv::FloatImage fimage(&image);

                // create feature mask
                nv::BitMap bmp(image.width, image.height);
                bmp.clearAll();
                const int w = image.width;
                const int h = image.height;
                int ytr = h;   //height of the transparent part
                for (int y = 0; y<h; ++y)
                    for (int x = 0; x<w; ++x)
                        if (fimage.pixel(3, x, y, 0) >= 0.5f) {
                            bmp.setBitAt(x, y);
                            if (y < ytr) ytr = y;
                        }


                // fill holes
                nv::fillVoronoi(&fimage, &bmp);

                // do blur passes
                for (int i = 0; i<8; ++i)
                    nv::fillBlur(&fimage, &bmp);

                nv::AutoPtr<nv::Image> img(fimage.createImage(0));

                inputOptions.setTextureLayout(nvtt::TextureType_2D, img->width, img->height);
                inputOptions.setMipmapData(img->pixels(), img->width, img->height);
            }
            else {
                inputOptions.setTextureLayout(nvtt::TextureType_2D, image.width, image.height);
                inputOptions.setMipmapData(image.pixels(), image.width, image.height);
            }
        }


        if (format == nvtt::Format_Unknown)
            format = alpha ? nvtt::Format_BC1a : nvtt::Format_BC1;


        if (wrapRepeat)
        {
            inputOptions.setWrapMode(nvtt::WrapMode_Repeat);
        }
        else
        {
            inputOptions.setWrapMode(nvtt::WrapMode_Clamp);
        }

        if (alpha)
        {
            inputOptions.setAlphaMode(nvtt::AlphaMode_Transparency);
        }
        else
        {
            inputOptions.setAlphaMode(nvtt::AlphaMode_None);
        }

        // IC: Do not enforce D3D9 restrictions anymore.
        // Block compressed textures with mipmaps must be powers of two.
        /*if (!noMipmaps && format != nvtt::Format_RGB)
        {
            //inputOptions.setRoundMode(nvtt::RoundMode_ToPreviousPowerOfTwo);
        }*/

        if (highPassMips) {
            inputOptions.setNormalMap(true);
            inputOptions.setConvertToNormalMap(false);
            inputOptions.setGamma(1.0f, 1.0f);
            inputOptions.setNormalizeMipmaps(false);
        }
        else if (linear)
        {
            setLinearMap(inputOptions);
        }
        else if (normal)
        {
            setNormalMap(inputOptions);
        }
        else if (color2normal)
        {
            setColorToNormalMap(inputOptions);
        }
        else
        {
            setColorMap(inputOptions);
        }

        if (noMipmaps)
        {
            inputOptions.setMipmapGeneration(false);
        }

        if (premultiplyAlpha)
        {
            //inputOptions.setPremultiplyAlpha(true);
            inputOptions.setAlphaMode(nvtt::AlphaMode_Premultiplied);
        }

        inputOptions.setMipmapFilter(mipmapFilter);
    }



    nvtt::CompressionOptions compressionOptions;
    compressionOptions.setFormat(format);

    //compressionOptions.setQuantization(/*color dithering*/true, /*alpha dithering*/false, /*binary alpha*/false);

    if (format == nvtt::Format_BC2) {
        // Dither alpha when using BC2.
        compressionOptions.setQuantization(/*color dithering*/false, /*alpha dithering*/true, /*binary alpha*/false);
    }
    else if (format == nvtt::Format_BC1a) {
        // Binary alpha when using BC1a.
        compressionOptions.setQuantization(/*color dithering*/false, /*alpha dithering*/true, /*binary alpha*/true, 127);
    }
    else if (format == nvtt::Format_RGBA)
    {
        if (luminance)
        {
            compressionOptions.setPixelFormat(8, 0xff, 0, 0, 0);
        }
        else {
            // @@ Edit this to choose the desired pixel format:
            // compressionOptions.setPixelType(nvtt::PixelType_Float);
            // compressionOptions.setPixelFormat(16, 16, 16, 16);
            // compressionOptions.setPixelType(nvtt::PixelType_UnsignedNorm);
            // compressionOptions.setPixelFormat(16, 0, 0, 0);

            //compressionOptions.setQuantization(/*color dithering*/true, /*alpha dithering*/false, /*binary alpha*/false);
            //compressionOptions.setPixelType(nvtt::PixelType_UnsignedNorm);
            //compressionOptions.setPixelFormat(5, 6, 5, 0);
            //compressionOptions.setPixelFormat(8, 8, 8, 8);

            // A4R4G4B4
            //compressionOptions.setPixelFormat(16, 0xF00, 0xF0, 0xF, 0xF000);

            //compressionOptions.setPixelFormat(32, 0xFF0000, 0xFF00, 0xFF, 0xFF000000);

            // R10B20G10A2
            //compressionOptions.setPixelFormat(10, 10, 10, 2);

            // DXGI_FORMAT_R11G11B10_FLOAT
            //compressionOptions.setPixelType(nvtt::PixelType_Float);
            //compressionOptions.setPixelFormat(11, 11, 10, 0);
        }
    }
    else if (format == nvtt::Format_BC6)
    {
        compressionOptions.setPixelType(nvtt::PixelType_UnsignedFloat);
    }

    if (fast)
    {
        compressionOptions.setQuality(nvtt::Quality_Fastest);
    }
    else
    {
        compressionOptions.setQuality(nvtt::Quality_Normal);
        //compressionOptions.setQuality(nvtt::Quality_Production);
        //compressionOptions.setQuality(nvtt::Quality_Highest);
    }

    if (bc1n)
    {
        compressionOptions.setColorWeights(1, 1, 0);
    }

    
    //compressionOptions.setColorWeights(0.2126, 0.7152, 0.0722);
    //compressionOptions.setColorWeights(0.299, 0.587, 0.114);
    //compressionOptions.setColorWeights(3, 4, 2);

    if (externalCompressor != NULL)
    {
        compressionOptions.setExternalCompressor(externalCompressor);
    }




    MyErrorHandler errorHandler;
    MyOutputHandler* outputHandler = 0;
    
    if (zstd)
        outputHandler = new ZstdOutputHandler(output.str());
    else
        outputHandler = new MyOutputHandler(output.str());

    if (outputHandler->stream->isError())
    {
        fprintf(stderr, "Error opening '%s' for writting\n", output.str());
        return EXIT_FAILURE;
    }

    nvtt::Context context;
    context.enableCudaAcceleration(!nocuda);

    if (!silent) 
    {
        printf("CUDA acceleration ");
        if (context.isCudaAccelerationEnabled())
        {
            printf("ENABLED\n\n");
        }
        else
        {
            printf("DISABLED\n\n");
        }
    }

    int outputSize = 0;
    if (useSurface) {
        outputSize = context.estimateSize(image, 1, compressionOptions);
    }
    else {
        outputSize = context.estimateSize(inputOptions, compressionOptions);
    }

    outputHandler->setTotal(outputSize);
    outputHandler->setDisplayProgress(!silent);

    nvtt::OutputOptions outputOptions;
    //outputOptions.setFileName(output);
    outputOptions.setOutputHandler(outputHandler);
    outputOptions.setErrorHandler(&errorHandler);

    if (ktx)
    {
        outputOptions.setContainer(nvtt::Container_KTX);
    }
    else
    {
	// Automatically use dds10 if compressing to BC6 or BC7
        if (format == nvtt::Format_BC6 || format == nvtt::Format_BC7) {
		dds10 = true;
	}

        if (dds10) {
        outputOptions.setContainer(nvtt::Container_DDS10);
    }
        else {
            outputOptions.setContainer(nvtt::Container_DDS);
        }
    }
    
    if (srgb) {
        outputOptions.setSrgbFlag(true);
    }

    // printf("Press ENTER.\n");
    // fflush(stdout);
    // getchar();

    nv::Timer timer;
    timer.start();

    if (useSurface) {
        if (!context.outputHeader(image, 1, compressionOptions, outputOptions)) {
            fprintf(stderr, "Error writing file header.\n");
            return EXIT_FAILURE;
        }
        if (!context.compress(image, 0, 0, compressionOptions, outputOptions)) {
            fprintf(stderr, "Error compressing file.\n");
            return EXIT_FAILURE;
        } 
    }
    else {
        if (!context.process(inputOptions, compressionOptions, outputOptions)) {
            return EXIT_FAILURE;
        }
    }

    //flush
    outputHandler->writeData(0, 0);

    timer.stop();

    if (!silent) {
        printf("\rtime taken: %.3f seconds\n", timer.elapsed());
    }

    return EXIT_SUCCESS;
}

