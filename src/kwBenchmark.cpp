#include "kwBenchmark.h"

#include <cmath>
#include <iomanip>
#include <iostream>


std::map<std::string, kw::BenchmarkRecord>
    g_benchmarkJournal;


kw::BenchmarkRecord&
    benchmarkJournal(const std::string& label)
{
    static std::map<std::string, kw::BenchmarkRecord>
        m_benchmarkJournal;

    return m_benchmarkJournal[label];
};


void
kw::BenchmarkRecord::print(const std::string& label, bool details)
{
    auto timePrecision = 3;
    auto timeUnits = "ms";

    std::cout << std::setprecision(timePrecision) << std::fixed;
    std::cout << "Benchmark for " << label << std::endl;

    if (m_n == 0)
    {
        std::cout << "    empty" << std::endl;
        std::cout << std::endl;
        return;
    }
    const auto avgTime = m_avgTime / m_n;
    const auto stdTime = std::sqrt(m_stdTime / m_n - avgTime * avgTime);
    std::cout << "    funCall: " << m_n << " times" << std::endl;
    std::cout << "    totTime: " << m_avgTime << timeUnits << std::endl;
    if (details)
    {
        std::cout << "    avgTime: " << avgTime << timeUnits << std::endl;
        std::cout << "    stdTime: " << stdTime << timeUnits << std::endl;
        std::cout << "    minTime: " << m_minTime << timeUnits << std::endl;
        std::cout << "    maxTime: " << m_maxTime << timeUnits << std::endl;
    }
    std::cout << std::endl;
}

void
kw::BenchmarkRecord::pause()
{
    if (!m_active)
    {
        std::cout << "pause: already deactivated" << std::endl;
        return;
    }

    const auto dt = (std::chrono::steady_clock::now() - m_last).count() * 1e-6;
    //const auto dt = (std::clock() - m_lastClock) * 1.0;

    m_avgTime += dt;
    m_stdTime += dt * dt;
    m_maxTime = m_n == 0 ? dt : std::max(m_maxTime, dt);
    m_minTime = m_n == 0 ? dt : std::min(m_minTime, dt);

    m_n += 1;

    m_active = false;
}

void
kw::BenchmarkRecord::resume()
{
    if (m_active)
    {
        std::cout << "resume: already active" << std::endl;
        return;
    }

    m_last = std::chrono::steady_clock::now();
    //m_lastClock = std::clock();

    m_active = true;
}
