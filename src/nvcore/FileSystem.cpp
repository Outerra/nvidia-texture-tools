// This code is in the public domain -- castano@gmail.com

#include "FileSystem.h"

#if NV_OS_WIN32
#define _CRT_NONSTDC_NO_WARNINGS // _chdir is defined deprecated, but that's a bug, chdir is deprecated, _chdir is *not*.
//#include <shlwapi.h> // PathFileExists
#include <windows.h> // GetFileAttributes
#include <direct.h> // _mkdir, _chdir
#elif NV_OS_XBOX
#include <Xtl.h>
#elif NV_OS_ORBIS
#include <fios2.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <stdio.h> // remove, unlink

using namespace nv;


bool FileSystem::exists(const char * path)
{
#if NV_OS_UNIX
	return access(path, F_OK|R_OK) == 0;
	//struct stat buf;
	//return stat(path, &buf) == 0;
#elif NV_OS_WIN32 || NV_OS_XBOX
    // PathFileExists requires linking to shlwapi.lib
    //return PathFileExists(path) != 0;
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
	if (FILE * fp = fopen(path, "r"))
	{
		fclose(fp);
		return true;
	}
	return false;
#endif
}

bool FileSystem::createDirectory(const char * path)
{
#if NV_OS_WIN32 || NV_OS_XBOX
    return CreateDirectoryA(path, NULL) != 0;
#elif NV_OS_ORBIS
    // not implemented
	return false;
#else
    return mkdir(path, 0777) != -1;
#endif
}

bool FileSystem::changeDirectory(const char * path)
{
#if NV_OS_WIN32
    return _chdir(path) != -1;
#elif NV_OS_XBOX
	// Xbox doesn't support Current Working Directory!
	return false;
#elif NV_OS_ORBIS
    // Orbis doesn't support Current Working Directory!
	return false;
#else
    return chdir(path) != -1;
#endif
}

bool FileSystem::removeFile(const char * path)
{
    // @@ Use unlink or remove?
    return remove(path) == 0;
}


#include "StdStream.h" // for fileOpen

bool FileSystem::copyFile(const char * src, const char * dst) {

    FILE * fsrc = fileOpen(src, "rb");
    if (fsrc == NULL) return false;
    defer{ fclose(fsrc); };

    FILE * fdst = fileOpen(dst, "wb");
    if (fdst == NULL) return false;
    defer{ fclose(fdst); };
    
    char buffer[1024];
    size_t n;

    while ((n = fread(buffer, sizeof(char), sizeof(buffer), fsrc)) > 0) {
        if (fwrite(buffer, sizeof(char), n, fdst) != n) {
            return false;
        }
    }
    
    return true;
}

bool nv::FileSystem::setFileModTime(const char* src, time_t imtime)
{
    FILETIME ftmod;
    uint64 ftx = (imtime * 10000000ULL) + 116444736000000000;
    ftmod.dwHighDateTime = uint32(ftx >> 32);
    ftmod.dwLowDateTime = uint32(ftx);

    HANDLE h = CreateFile(src, GENERIC_WRITE, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (h == INVALID_HANDLE_VALUE)
        return false;

    BOOL r = SetFileTime(h, NULL, NULL, &ftmod);

    CloseHandle(h);

    return r != 0;
}


