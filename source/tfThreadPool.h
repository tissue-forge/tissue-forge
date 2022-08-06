/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

/*
 Based on Magnum example
 
 Original authors — credit is appreciated but not required:
 
 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
 Vladimír Vondruš <mosra@centrum.cz>
 2019 — Nghia Truong <nghiatruong.vn@gmail.com>
 
 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.
 
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 */

#ifndef _SOURCE_TFTHREADPOOL_H_
#define _SOURCE_TFTHREADPOOL_H_

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <Magnum/Math/Functions.h>


namespace TissueForge {


    /* This is a very simple threadpool implementation, for demonstration purpose
    only. Using tbb::parallel_for from Intel TBB yields higher performance --
    see TaskScheduler.h. */
    class ThreadPool {
    public:
        ThreadPool() {
            const int maxNumThreads = std::thread::hardware_concurrency();
            std::size_t nWorkers = maxNumThreads > 1 ? maxNumThreads - 1 : 0;
            
            _threadTaskReady.resize(nWorkers, 0);
            _tasks.resize(nWorkers + 1);
            
            for(std::size_t threadIdx = 0; threadIdx < nWorkers; ++threadIdx) {
                _workerThreads.emplace_back([threadIdx, this] {
                    for(;;) {
                        {
                            std::unique_lock<std::mutex> lock(_taskMutex);
                            _condition.wait(lock, [threadIdx, this] {
                                return _bStop || _threadTaskReady[threadIdx] == 1;
                            });
                            if(_bStop && !_threadTaskReady[threadIdx]) return;
                        }
                        
                        _tasks[threadIdx](); /* run task */
                        
                        /* Set task ready to 0, thus this thread will not do
                        its computation more than once */
                        _threadTaskReady[threadIdx] = 0;
                        
                        /* Decrease the busy thread counter */
                        _numBusyThreads.fetch_add(-1);
                    }
                });
            }
        }
        
        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(_taskMutex);
                _bStop = true;
            }
            
            _condition.notify_all();
            for(std::thread& worker: _workerThreads) worker.join();
        }
        
        void parallel_for(std::size_t size, std::function<void(std::size_t)>&& func) {
            const auto nWorkers = _workerThreads.size();
            if(nWorkers > 0) {
                _numBusyThreads = int(nWorkers);
                
                const std::size_t chunkSize = std::size_t(Magnum::Math::ceil(float(size)/ float(nWorkers + 1)));
                for(std::size_t threadIdx = 0; threadIdx < nWorkers + 1; ++threadIdx) {
                    const std::size_t chunkStart = threadIdx * chunkSize;
                    const std::size_t chunkEnd = Magnum::Math::min(chunkStart + chunkSize, size);
                    
                    /* Must copy func into local lambda's variable */
                    _tasks[threadIdx] = [chunkStart, chunkEnd, task = func] {
                        for(uint64_t idx = chunkStart; idx < chunkEnd; ++idx) {
                            task(idx);
                        }
                    };
                }
                
                /* Wake up worker threads */
                {
                    std::unique_lock<std::mutex> lock(_taskMutex);
                    for(std::size_t threadIdx = 0; threadIdx < _threadTaskReady.size(); ++threadIdx)
                        _threadTaskReady[threadIdx] = 1;
                }
                _condition.notify_all();
                
                /* Handle last chunk in this thread */
                _tasks.back()();
                
                /* Wait until all worker threads finish */
                while(_numBusyThreads.load() > 0) {}
                
            } else for(std::size_t idx = 0; idx < size; ++idx)
                func(idx);
        }
        
        static ThreadPool& getUniqueInstance() {
            static ThreadPool threadPool;
            return threadPool;
        }
        
        static int hardwareThreadSize() {
            return std::thread::hardware_concurrency();
        }
        
        static int size() {
            return getUniqueInstance()._workerThreads.size();
        }
        
    private:
        std::atomic<int> _numBusyThreads{0};
        /* Do not use std::vector<bool>: it's not threadsafe */
        std::vector<int> _threadTaskReady;
        std::vector<std::thread> _workerThreads;
        
        std::vector<std::function<void()>> _tasks;
        std::mutex _taskMutex;
        std::condition_variable _condition;
        bool _bStop = false;
    };
}

#endif // _SOURCE_TFTHREADPOOL_H_