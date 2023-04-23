#include <iostream>

#include "docopt.h"

#define KW_BENCHMARK_ON
#include "kwBenchmark.h"

#include "kwFd1d.h"
#include "kwTime.h"


namespace kw {

constexpr char usage[] = R"(
kwinto - Financial Analytics

Usage:
  kwinto benchmark [-n <num>] [-k <num>] [--nThreads <num>] [--xThreads <num>]

Options:
  -n <num>          Batch size [default: 256]
  -k <num>          Repeatition count [default: 4]
  --nThreads <num>  Thread block size in n-dim [default: 1]
  --xThreads <num>  Thread block size in x-dim [default: 128]
  -h --help         Show this screen
  --version         Show version
)";

}

int main(int argc, char** argv)
{
    std::map<std::string, docopt::value> args;
    args = docopt::docopt(
        kw::usage,
        { argv + 1, argv + argc },
        true,
        "kwinto v0.1.0");

    for (auto const& arg : args) {
        std::cout << arg.first << ": " << arg.second << std::endl;
    }
    std::cout << std::endl;

    auto n = args["-n"].asLong();
    auto k = args["-k"].asLong();

    std::cout << "Portfolio:" << std::endl;
    std::cout << "    options:  " << n << std::endl;
    std::cout << "    exercise: american" << std::endl;
    std::cout << std::endl;

    {
        kw::Option<float> euroPut;
        euroPut.e = false;
        euroPut.k = 100;
        euroPut.r = 0.05;
        euroPut.t = 1.0;
        euroPut.z = 0.2;
        euroPut.w = kw::kParity::Put;

        std::vector<kw::Option<float>> assets;
        for (auto i = 0; i < n; ++i)
            assets.push_back(euroPut);


        kw::Fd1dConfig config;
        config.pdeCount = assets.size();
        config.tDim = config.xDim = 512;
        config.nThreads = args["--nThreads"].asLong();
        config.xThreads = args["--xThreads"].asLong();


        {
            kw::Fd1d<float> pricer;
            pricer.allocate(config);

            std::vector<kw::Fd1dPde<float>> pdes;
            kw::Fd1dPdeFor(assets, config, pdes);

            for (auto i = 0; i < k; ++i)
            {
                KW_BENCHMARK_RESUME("Fd1d<float>::solve");

                if (auto error = pricer.solve(pdes); !error.empty())
                {
                    std::cerr << error;
                    return 1;
                }

                KW_BENCHMARK_PAUSE("Fd1d<float>::solve");
            }

            pricer.free();
        }
        KW_BENCHMARK_PRINT("Fd1d<float>::solve");

        {
            kw::Fd1d_Gpu<float> pricer;
            pricer.allocate(config);

            std::vector<kw::Fd1dPde<float>> pdes;
            kw::Fd1dPdeFor(assets, config, pdes);

            for (auto i = 0; i < k; ++i)
            {
                KW_BENCHMARK_RESUME("Fd1d_Gpu<float>::solve");

                if (auto error = pricer.solve(pdes); !error.empty())
                {
                    std::cerr << error;
                    return 1;
                }

                KW_BENCHMARK_PAUSE("Fd1d_Gpu<float>::solve");
            }

            pricer.free();
        }
        KW_BENCHMARK_PRINT("Fd1d_Gpu<float>::solve");
    }


    {
        kw::Option<double> euroPut;
        euroPut.e = false;
        euroPut.k = 100;
        euroPut.r = 0.05;
        euroPut.t = 1.0;
        euroPut.z = 0.2;
        euroPut.w = kw::kParity::Put;

        std::vector<kw::Option<double>> assets;
        for (auto i = 0; i < n; ++i)
            assets.push_back(euroPut);

        kw::Fd1dConfig config;
        config.pdeCount = assets.size();
        config.tDim = config.xDim = 512;
        config.nThreads = args["--nThreads"].asLong();
        config.xThreads = args["--xThreads"].asLong();

        {
            kw::Fd1d<double> pricer;
            pricer.allocate(config);

            std::vector<kw::Fd1dPde<double>> pdes;
            kw::Fd1dPdeFor(assets, config, pdes);


            for (auto i = 0; i < k; ++i)
            {
                KW_BENCHMARK_RESUME("Fd1d<double>::solve");

                if (auto error = pricer.solve(pdes); !error.empty())
                {
                    std::cerr << error;
                    return 1;
                }

                KW_BENCHMARK_PAUSE("Fd1d<double>::solve");
            }

            pricer.free();
        }
        KW_BENCHMARK_PRINT("Fd1d<double>::solve");

        {
            kw::Fd1d_Gpu<double> pricer;
            pricer.allocate(config);

            std::vector<kw::Fd1dPde<double>> pdes;
            kw::Fd1dPdeFor(assets, config, pdes);


            for (auto i = 0; i < k; ++i)
            {
                KW_BENCHMARK_RESUME("Fd1d_Gpu<double>::solve");

                if (auto error = pricer.solve(pdes); !error.empty())
                {
                    std::cerr << error;
                    return 1;
                }

                KW_BENCHMARK_PAUSE("Fd1d_Gpu<double>::solve");
            }

            pricer.free();
        }
        KW_BENCHMARK_PRINT("Fd1d_Gpu<double>::solve");
    }

    return 0;
}

