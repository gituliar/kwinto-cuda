#pragma once

#include <chrono>
#include <map>
#include <string>

namespace kw
{

class BenchmarkRecord
{
private:
    size_t  m_n = 0;

    double  m_avgTime = 0;
    double  m_maxTime = 0;
    double  m_minTime = 0;
    double  m_stdTime = 0;

    bool    m_active = false;
    std::chrono::time_point<std::chrono::steady_clock>
            m_last;

public:
    void
        print(const std::string& label, bool details = true);
    void
        pause();
    void
        resume();
};

}

kw::BenchmarkRecord&
    benchmarkJournal(const std::string& label);

#ifdef KW_BENCHMARK_ON

#define KW_BENCHMARK_PAUSE(label) benchmarkJournal(label).pause();
#define KW_BENCHMARK_RESUME(label) benchmarkJournal(label).resume();

#define KW_BENCHMARK_PRINT(label) benchmarkJournal(label).print(label);
#define KW_BENCHMARK_PRINT_SHORT(label) benchmarkJournal(label).print(label, false);

#else

#define KW_BENCHMARK_PAUSE(label)
#define KW_BENCHMARK_RESUME(label)

#define KW_BENCHMARK_PRINT(label)
#define KW_BENCHMARK_PRINT_SHORT(label)


#endif
