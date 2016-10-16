#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <chrono>
#include <typeinfo>
#include <string.h>
#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#define USE_AVX2 1

class Matcher
{
public:
    virtual bool match (const char * p) const = 0;
};

class NullMatcher : public Matcher
{
    bool match (const char * p) const
    {
        return false;
    }
};

class VectorLinearMatcher : public Matcher
{
    std::vector<uint32_t> values;

    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        for (uint32_t x : values) {
            if (x == v) return true;
        }
        return false;
    }

public:
    VectorLinearMatcher ()
    {
        values.push_back ('/PIS');
        values.push_back ('IVNI');
        values.push_back (' KCA');
        values.push_back ('CNAC');
        values.push_back (' EYB');
        values.push_back ('CARP');
        values.push_back ('IGER');
        values.push_back ('ITPO');
        values.push_back ('OFNI');
        values.push_back ('ADPU');
        values.push_back ('SBUS');
        values.push_back ('ITON');
        values.push_back ('SSEM');
        values.push_back ('EFER');
        values.push_back ('LBUP');
    }
};

class VectorFunctionalLinearMatcher : public Matcher
{
    std::vector<uint32_t> values;

    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        return std::any_of (values.cbegin (), values.cend (), [v](uint32_t x) { return x == v; });
    }

public:
    VectorFunctionalLinearMatcher ()
    {
        values.push_back ('/PIS');
        values.push_back ('IVNI');
        values.push_back (' KCA');
        values.push_back ('CNAC');
        values.push_back (' EYB');
        values.push_back ('CARP');
        values.push_back ('IGER');
        values.push_back ('ITPO');
        values.push_back ('OFNI');
        values.push_back ('ADPU');
        values.push_back ('SBUS');
        values.push_back ('ITON');
        values.push_back ('SSEM');
        values.push_back ('EFER');
        values.push_back ('LBUP');
    }
};

class VectorBinaryMatcher : public Matcher
{
    std::vector<uint32_t> values;

    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        return binary_search (values.cbegin (), values.cend (), v);
    }

public:
    VectorBinaryMatcher ()
    {
        values.push_back ('/PIS');
        values.push_back ('IVNI');
        values.push_back (' KCA');
        values.push_back ('CNAC');
        values.push_back (' EYB');
        values.push_back ('CARP');
        values.push_back ('IGER');
        values.push_back ('ITPO');
        values.push_back ('OFNI');
        values.push_back ('ADPU');
        values.push_back ('SBUS');
        values.push_back ('ITON');
        values.push_back ('SSEM');
        values.push_back ('EFER');
        values.push_back ('LBUP');
        sort (values.begin(), values.end());
    }
};

class LinearMatcher : public Matcher
{
    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        if (v == '/PIS') return true;
        if (v == 'IVNI') return true;
        if (v == ' KCA') return true;
        if (v == 'CNAC') return true;
        if (v == ' EYB') return true;
        if (v == 'CARP') return true;
        if (v == 'IGER') return true;
        if (v == 'ITPO') return true;
        if (v == 'OFNI') return true;
        if (v == 'ADPU') return true;
        if (v == 'SBUS') return true;
        if (v == 'ITON') return true;
        if (v == 'SSEM') return true;
        if (v == 'EFER') return true;
        if (v == 'LBUP') return true;
        return false;
    }
};

class BinaryMatcher : public Matcher
{
    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;

#define IF_LESS(x) \
        if (v == x) return true;\
        if (v < x)

        IF_LESS ('IGER') {
            IF_LESS ('ADPU') {
                IF_LESS (' KCA') {
                    return v == ' EYB';
                } else {
                    return v == '/PIS';
                }
            } else {
                IF_LESS ('CNAC') {
                    return v == 'CARP';
                } else {
                    return v == 'EFER';
                }
            }
        } else {
            IF_LESS ('LBUP') {
                IF_LESS ('ITPO') {
                    return v == 'ITON';
                } else {
                    return v == 'IVNI';
                }
            } else {
                IF_LESS ('SBUS') {
                    return v == 'OFNI';
                } else {
                    return v == 'SSEM';
                }
            }
        }
#undef IF_LESS
    }
};

class TreeSetMatcher : public Matcher
{
    std::set<uint32_t> values;

    bool match (const char * p) const
    {
        return values.find (*(const uint32_t*) p) != values.end ();
    }

public:
    TreeSetMatcher ()
    {
        values.insert ('/PIS');
        values.insert ('IVNI');
        values.insert (' KCA');
        values.insert ('CNAC');
        values.insert (' EYB');
        values.insert ('CARP');
        values.insert ('IGER');
        values.insert ('ITPO');
        values.insert ('OFNI');
        values.insert ('ADPU');
        values.insert ('SBUS');
        values.insert ('ITON');
        values.insert ('SSEM');
        values.insert ('EFER');
        values.insert ('LBUP');
    }
};

class HashSetMatcher : public Matcher
{
    std::unordered_set<uint32_t> values;

    bool match (const char * p) const
    {
        return values.find (*(const uint32_t*) p) != values.end ();
    }

public:
    HashSetMatcher ()
    {
        values.insert ('/PIS');
        values.insert ('IVNI');
        values.insert (' KCA');
        values.insert ('CNAC');
        values.insert (' EYB');
        values.insert ('CARP');
        values.insert ('IGER');
        values.insert ('ITPO');
        values.insert ('OFNI');
        values.insert ('ADPU');
        values.insert ('SBUS');
        values.insert ('ITON');
        values.insert ('SSEM');
        values.insert ('EFER');
        values.insert ('LBUP');
    }
};

class SSEMatcher : public Matcher
{
    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        __m128i w = _mm_setr_epi32 (v, v, v, v);

#define CMP(x) \
        if (_mm_movemask_epi8 (_mm_cmpeq_epi32 (w, x))) return true;

        CMP (_mm_setr_epi32 ('/PIS', 'IVNI', ' KCA', 'CNAC'));
        CMP (_mm_setr_epi32 (' EYB', 'CARP', 'IGER', 'ITPO'));
        CMP (_mm_setr_epi32 ('OFNI', 'ADPU', 'SBUS', 'ITON'));
        CMP (_mm_setr_epi32 ('SSEM', 'EFER', 'LBUP', 'LBUP'));
#undef CMP
        return false;
    }
};

class SSEMatcher2 : public Matcher
{
    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        __m128i w = _mm_setr_epi32 (v, v, v, v);
        __m128i d = _mm_setzero_si128 ();

#define CMP(x) \
        d = _mm_or_si128 (d, _mm_cmpeq_epi32 (w, x));

        CMP (_mm_setr_epi32 ('/PIS', 'IVNI', ' KCA', 'CNAC'));
        CMP (_mm_setr_epi32 (' EYB', 'CARP', 'IGER', 'ITPO'));
        CMP (_mm_setr_epi32 ('OFNI', 'ADPU', 'SBUS', 'ITON'));
        CMP (_mm_setr_epi32 ('SSEM', 'EFER', 'LBUP', 'LBUP'));
#undef CMP
        return _mm_movemask_epi8 (d) != 0;
    }
};

#if USE_AVX2
class AVX2Matcher : public Matcher
{
    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        __m256i w = _mm256_setr_epi32 (v, v, v, v, v, v, v, v);
        __m256i d = _mm256_setzero_si256 ();

#define CMP(x) \
        d = _mm256_or_si256 (d, _mm256_cmpeq_epi32 (w, x));

        CMP (_mm256_setr_epi32 ('/PIS', 'IVNI', ' KCA', 'CNAC', ' EYB', 'CARP', 'IGER', 'ITPO'));
        CMP (_mm256_setr_epi32 ('OFNI', 'ADPU', 'SBUS', 'ITON', 'SSEM', 'EFER', 'LBUP', 'LBUP'));
#undef CMP
        return _mm256_movemask_epi8 (d) != 0;
    }
};
#endif

class CustomHashMatcher : public Matcher
{
    uint32_t values [16];

    inline uint32_t hash (uint32_t v) const
    {
        return (v * 239012) >> 28;
    }

    bool match (const char * p) const
    {
        uint32_t v = *(const uint32_t*) p;
        return values [hash (v)] == v;
    }

    void insert (uint32_t v)
    {
        uint32_t h = hash (v);
        if (values [h]) exit (1);
        values [h] = v;
    }

public:
    CustomHashMatcher ()
    {
        memset (values, 0, sizeof (values));
        insert ('/PIS');
        insert ('IVNI');
        insert (' KCA');
        insert ('CNAC');
        insert (' EYB');
        insert ('CARP');
        insert ('IGER');
        insert ('ITPO');
        insert ('OFNI');
        insert ('ADPU');
        insert ('SBUS');
        insert ('ITON');
        insert ('SSEM');
        insert ('EFER');
        insert ('LBUP');
        if (values [0] == 0) exit (1);
    }
};

class CustomPortableHashMatcher : public Matcher
{
    uint32_t values [16];

    inline uint32_t hash (uint32_t v) const
    {
        return (v * 239012) >> 28;
//        return (v * 93564) >> 28;
    }

    bool match (const char * p) const
    {
        uint32_t v;
        memcpy (&v, p, sizeof (v));
        v = ntohl (v);
        return values [hash (v)] == v;
    }

    void insert (const char * p)
    {
        uint32_t v;
        memcpy (&v, p, sizeof (v));
        v = ntohl (v);
        uint32_t h = hash (v);
        printf ("%c%c%c%c: %d\n", v>>24, v>>16,v>>8,v, h);
        if (values [h]) printf ("Jopa\n");// exit (1);
        values [h] = v;
    }

public:
    CustomPortableHashMatcher ()
    {
        memset (values, 0, sizeof (values));
        insert ("SIP/");
        insert ("INVI");
        insert ("ACK ");
        insert ("CANC");
        insert ("BYE ");
        insert ("PRAC");
        insert ("REGI");
        insert ("OPTI");
        insert ("INFO");
        insert ("UPDA");
        insert ("SUBS");
        insert ("NOTI");
        insert ("MESS");
        insert ("REFE");
        insert ("PUBL");
        if (values [0] == 0) exit (1);
    }
};

std::vector<const char*> goodtests = {
    "PUBLQWERTYUIOI",
    "INFOQWERTY",
    "SIP/htriuhftier",
    "UPDAZXCVBNMN",
    "REGIdede"
};

std::vector<const char*> badtests = {
    "ABCD123456",
    "123456",
    "4376834783974",
    "fe3htr3rjhtfkjer",
    "44hkjhjkghkh",
};


void test (const Matcher &matcher, const std::vector<const char*>& tests, const char* name)
{
    int REP = 10000000;
    int sum = 0;

    auto t0 = std::chrono::system_clock::now ();
    for (int i = 0; i < REP; i++) {
        for (const char * str : tests) {
            if (matcher.match (str)) ++sum;
        }
    }
    auto t1 = std::chrono::system_clock::now ();
    std::chrono::duration<double, std::nano> nsdur = t1 - t0;

    std::cout << std::left <<
        std::setw (5) << name << 
        std::setw (40) << typeid (matcher).name () << ": "
        << std::right << std::setw (10) << sum << ": "
        << std::fixed << std::setw (6) << std::setprecision (2) << nsdur.count () / (REP * tests.size())
        << " ns\n";

}

void test (const Matcher &matcher)
{
    test (matcher, goodtests, "good");
    test (matcher, badtests, "bad");
}

int main (void)
{
    test (NullMatcher ());
    test (NullMatcher ());
    test (VectorLinearMatcher ());
    test (VectorFunctionalLinearMatcher ());
    test (VectorBinaryMatcher ());
    test (LinearMatcher ());
    test (BinaryMatcher ());
    test (TreeSetMatcher ());
    test (HashSetMatcher ());
    test (SSEMatcher ());
    test (SSEMatcher2 ());
#if USE_AVX2
    test (AVX2Matcher ());
#endif
    test (CustomHashMatcher ());
    test (CustomPortableHashMatcher ());
    return 0;
}
