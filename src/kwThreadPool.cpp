#include "kwThreadPool.h"

#include <iostream>

//#include "kwUtils/logging.h"



kw::ThreadPool::ThreadPool()
{
    auto thread_count = std::thread::hardware_concurrency();
    //spdlog::info("ThreadPool::ThreadPool : {} threads", thread_count);

    for (int i = 0; i < thread_count; ++i)
    {
        m_pool.push_back(std::thread(&ThreadPool::loop_thread, this));
    }

    m_active_jobs = 0;
    m_terminate = false;
};


kw::ThreadPool::~ThreadPool()
{
    m_terminate = true;

    m_queue_cv.notify_all();
    for (auto& thread : m_pool)
    {
        thread.join();
    }

};

kw::ThreadPool&
kw::ThreadPool::instance()
{
    static ThreadPool instance;
    return instance;
};

void
kw::ThreadPool::add_job(std::function<void()> job)
{
    {
        std::unique_lock<std::mutex> lock(m_queue_mutex);

        m_jobs.push(std::move(job));
    }

    m_queue_cv.notify_one();
}

void
kw::ThreadPool::stop() {};

void
kw::ThreadPool::wait()
{
    for (;;)
    {
        std::unique_lock<std::mutex> lock(m_wait_mutex);

        m_wait_cv.wait(lock, [this]()
        {
            return m_jobs.empty() && m_active_jobs == 0;
        });

        break;
    };
};


void
kw::ThreadPool::loop_thread()
{
    for (;;)
    {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);

            m_queue_cv.wait(lock, [this]()
            {
                return !m_jobs.empty() || m_terminate;
            });

            if (m_terminate)
                break;

            job = m_jobs.front();
            m_jobs.pop();
        }

        {
            std::unique_lock<std::mutex> lock(m_wait_mutex);
            m_active_jobs += 1;
        }

        try
        {
            job();
        }
        catch (const std::exception& ex)
        {
            //spdlog::error("ThreadPool::loop_thread : {}", ex.what());
            std::cerr << "ThreadPool::loop_thread : " << ex.what() << std::endl;
        }
        catch (...)
        {
            //spdlog::error("ThreadPool::loop_thread : unhandled exception");
            std::cerr << "ThreadPool::loop_thread : unhandled exception" << std::endl;
        }

        // notify 'wait' that we are done
        {
            std::unique_lock<std::mutex> lock(m_wait_mutex);
            m_active_jobs -= 1;
        }
        m_wait_cv.notify_all();
    }
};
