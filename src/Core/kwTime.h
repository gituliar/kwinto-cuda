#pragma once

#include <chrono>

namespace kw
{

class Timer
{
private:
    std::chrono::time_point<std::chrono::steady_clock>
        m_first;
    std::chrono::time_point<std::chrono::steady_clock>
        m_last;

public:
    Timer() :
        m_first{ std::chrono::steady_clock::now() }
    {
        m_last = m_first;
    };

    double
        elapsed_time_sec()
    {
        m_last = std::chrono::steady_clock::now();

        return (m_last - m_first).count() * 1e-9;
    };

    void
        reset()
    {
        m_first = m_last = std::chrono::steady_clock::now();
    }
};

}
