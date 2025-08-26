#pragma once
#include <chrono>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <string>

// RAII timer with nvtx ranges
class Timer {
public:
  Timer(std::string name) : m_name(name) {
    m_time0 = std::chrono::high_resolution_clock::now();
    nvtxRangePush(name.c_str());
  }

  ~Timer() {
    auto time1 = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(time1 - m_time0);
    std::cout << "Timing for " << m_name << ": " << time_span.count() << "s\n";
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_time0;
  std::string m_name;

};
