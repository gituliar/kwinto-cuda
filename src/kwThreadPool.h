#pragma once

#include <chrono>

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

//#include "kwUtils/logging.h"

using namespace std::chrono_literals;

namespace kw
{

    using Task = std::function<void()>;

    class ThreadPool
    {
    private:
        size_t
            m_active_jobs;
        bool
            m_terminate;

        std::queue<Task>
            m_jobs;

        std::vector<std::thread>
            m_pool;

        std::mutex
            m_queue_mutex;
        std::condition_variable
            m_queue_cv;

        std::mutex
            m_wait_mutex;
        std::condition_variable
            m_wait_cv;


    public:
        ThreadPool(ThreadPool const&) = delete;
        ~ThreadPool();

        void
            operator=(ThreadPool const&) = delete;

        static ThreadPool&
            instance();

        void
            add_job(std::function<void()> job);

        size_t
            size() const { return m_pool.size(); };
        void
            stop();
        void
            wait();


    private:
        ThreadPool();

        void
            loop_thread();
    };



    template<class ResultType>
    class ThreadQueue
    {
    private:
        std::queue<std::packaged_task<ResultType()>>
            m_taskQueue;
        std::vector<std::thread>
            m_threadPool;
        std::queue<std::future<ResultType>>
            m_resultQueue;
        size_t
            m_resultQueueCapacity;

        size_t
            m_runningTaskCount;

        std::condition_variable
            m_cv;
        std::mutex
            m_mutex;

        bool
            m_terminated;

    public:
        ThreadQueue() :
            ThreadQueue(2 * std::thread::hardware_concurrency()) { };

        ThreadQueue(size_t resultQueueCapacity) :
            ThreadQueue(resultQueueCapacity, std::thread::hardware_concurrency()) { };

        ThreadQueue(size_t resultQueueCapacity, size_t threadCount) :
            m_resultQueueCapacity{ resultQueueCapacity }, m_runningTaskCount{ 0 }, m_terminated{ false }
        {
            //spdlog::info("ThreadQueue::ThreadQueue : {} threads", threadCount);

            m_threadPool.reserve(threadCount);
            for (int i = 0; i < threadCount; ++i)
            {
                m_threadPool.push_back(std::thread(&ThreadQueue::worker, this));
            }
        };

        ThreadQueue(ThreadQueue const&) = delete;

        ~ThreadQueue()
        {
            terminate();

            for (auto& thread : m_threadPool)
            {
                thread.join();
            }
        };


        bool
            pop(std::future<ResultType>& value)
        {
            {
                std::unique_lock lock(m_mutex);

                while (!m_terminated && m_resultQueue.empty() && (m_runningTaskCount > 0 || !m_taskQueue.empty()))
                    m_cv.wait(lock);

                if (m_terminated || m_resultQueue.empty())
                    return false;

                value = std::move(m_resultQueue.front());
                m_resultQueue.pop();

                //spdlog::debug("ThreadQueue::pop : {} running tasks", m_runningTaskCount);
                //spdlog::debug("ThreadQueue::pop : {} results to pop", m_resultQueue.size());
                //spdlog::debug("ThreadQueue::pop : {} tasks to process", m_taskQueue.size());
            }

            m_cv.notify_one();

            return true;
        };

        void
            push(std::function<ResultType()> func)
        {
            push(std::packaged_task<ResultType()>(func));
        };

        void
            serve_forever()
        {
            std::unique_lock lock(m_mutex);

            while (!m_terminated)
                m_cv.wait(lock);
        };

        void
            terminate()
        {
            {
                std::scoped_lock lock(m_mutex);
                m_terminated = true;
            }

            m_cv.notify_all();
        };

        void
            wait()
        {
            std::unique_lock lock(m_mutex);

            while (!m_terminated && (!m_taskQueue.empty() || m_runningTaskCount > 0))
                m_cv.wait(lock);
        };

    private:
        void
            push(std::packaged_task<ResultType()> task)
        {
            {
                std::scoped_lock lock(m_mutex);

                m_taskQueue.push(std::move(task));
            }

            m_cv.notify_all();
        };


        void
            worker()
        {
            for (;;)
            {
                std::packaged_task<ResultType()> task;

                {
                    std::unique_lock lock(m_mutex);

                    while (!m_terminated && (m_taskQueue.empty() || (m_resultQueue.size() > m_resultQueueCapacity)))
                        m_cv.wait(lock);

                    if (m_terminated)
                        return;

                    if (m_resultQueue.size() > m_resultQueueCapacity)
                        continue;


                    task = std::move(m_taskQueue.front());
                    // TODO: handle returned error
                    m_taskQueue.pop();

                    if (m_resultQueueCapacity > 0)
                    {
                        m_resultQueue.push(task.get_future());
                    }

                    m_runningTaskCount += 1;
                };

                try
                {
                    task();
                }
                catch (const std::exception& ex)
                {
                    //spdlog::error("ThreadQueue::worker : {}", ex.what());
                }
                catch (...)
                {
                    //spdlog::error("ThreadQueue::worker : unhandled exception");
                }

                {
                    std::scoped_lock lock(m_mutex);

                    m_runningTaskCount -= 1;
                }

                m_cv.notify_all();
            }
        }
    };

}
