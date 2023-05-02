#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

#include "docopt.h"

#define KW_BENCHMARK_ON
#include "kwBenchmark.h"

#include "kwFd1d.h"
#include "kwTime.h"
#include "kwPortfolio.h"


namespace kw {

constexpr char usage[] = R"(
kwinto - Analytics for Options Trading

Usage:
    kwinto bench [options] <portfolio>

Arguments:
    <portfolio>     Read options data from <portfolio> CSV file

Options:
    --amer          Include american options
    -b <num>        Price <num> options per batch [default: 128]
    --call          Include call options
    --cpu32         Run single-precision benchmark on CPU 
    --cpu64         Run double-precision benchmark on CPU
    --cuda-n <num>  Run <num> threads per CUDA block over the n-dimension [default: 8]
    --cuda-x <num>  Run <num> threads per CUDA block over the x-dimension [default: 128]
    -e <num>        Reject option prices less than <num> from RRMS error stats [default: 0.5]
    --euro          Include european options
    --gpu32         Run single-precision benchmark on GPU
    --gpu64         Run double-precision benchmark on GPU
    -n <num>        Run <num> batches (when 0 run all) [default: 0]
    --put           Include put options
    -t <num>        Use <num> points for t-grid [default: 1024]
    -v              Show extra details, be verbose
    -x <num>        Use <num> points for x-grid [default: 1024]

    -h --help       Print this screen
    --version       Print Kwinto version
)";

}

using Args = std::map<std::string, docopt::value>;



kw::Error
    printGpuInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, i);

        std::cout << "Devices Info #" << i <<  std::endl;
        std::cout << "    Name:                 " << device.name << std::endl;
        std::cout << "    Integrated:           " << device.integrated << std::endl;
        std::cout << std::endl;

        std::cout << "    Total SM:             " <<
            device.multiProcessorCount << std::endl;
        std::cout << "    Clock Rate:           " <<
            device.clockRate / 1e3 << " MHz" << std::endl;
        //std::cout << "    Warp size (threads):  " <<
        //    device.warpSize << std::endl;
        std::cout << "    32-bit Regs (per SM): " <<
            device.regsPerMultiprocessor << std::endl;
        std::cout << "    Max Blocks (per SM):  " <<
            device.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "    Max Threads (per SM): " <<
            device.maxThreadsPerMultiProcessor << std::endl;
        std::cout << std::endl;

        std::cout << "    Total Memory:         " <<
            device.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "    Memory Clock Rate:    " <<
            device.memoryClockRate / 1e3 << " MHz" << std::endl;
        std::cout << "    Memory Bus Width:     " <<
            device.memoryBusWidth << " bits" << std::endl;
        std::cout << "    Peak Bandwidth:       " <<
            2.0 * device.memoryClockRate * (device.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
        std::cout << std::endl;
    }

    return "";
}


template<typename Real, typename Pricer>
kw::Error
    benchPortfolio(
        const kw::Portfolio& portfolio,
        kw::Fd1dConfig config,
        size_t batchSize,
        size_t batchCount,
        double tolerance,
        const std::string& label)
{
    Pricer pricer;

    if (batchSize == 0)
        batchSize = portfolio.size();
    config.pdeCount = batchSize;

    if (batchCount == 0)
        batchCount = (portfolio.size() + batchSize - 1) / batchSize;

    if (auto error = pricer.allocate(config); !error.empty())
        return "benchPortfolio: " + error;

    std::vector<std::vector<kw::Option>> batches(1);
    {
        auto assets = std::ref(batches[0]);
        for (const auto& [asset, _] : portfolio)
        {
            if (assets.get().size() == batchSize)
                assets = batches.emplace_back();
            assets.get().push_back(asset);
        }
    }

    double absDiffSum1 = 0, absDiffSum2 = 0, relDiffSum1 = 0, relDiffSum2 = 0;
    size_t absDiffSize = 0, relDiffSize = 0;
    for (auto i = 0; i < batchCount; ++i)
    {
        const auto& assets = batches[i];

        std::vector<kw::Fd1dPde<Real>> pdes;

        // 1. Init
        if (auto error = kw::Fd1dPdeFor(assets, config, pdes); !error.empty())
            return "benchPortfolio: " + error;

        // 2. Solve
        if (assets.size() == batchSize)
        {
            KW_BENCHMARK_RESUME(label);
        }

        if (auto error = pricer.solve(pdes); !error.empty())
            return "benchPortfolio: " + error;

        if (assets.size() == batchSize)
        {
            KW_BENCHMARK_PAUSE(label);
        }

        // 3. Collect statistics
        for (auto j = 0; j < assets.size(); ++j)
        {
            const auto& asset = assets[j];
            const auto& price = portfolio.at(asset);

            Real got;
            if (auto error = pricer.value(j, asset.s, got); !error.empty())
            {
                std::cerr << error << std::endl;
                continue;
            }

            if (price >= tolerance)
            {
                double absDiff = std::abs(price - got);
                double relDiff = absDiff / price;

                absDiffSum1 += absDiff;
                absDiffSum2 += absDiff * absDiff;
                absDiffSize++;

                relDiffSum1 += relDiff;
                relDiffSum2 += relDiff * relDiff;
                relDiffSize++;
            }
        }
    }

    {
        auto mae = absDiffSum1 / absDiffSize;
        auto rmse = std::sqrt(absDiffSum2 / absDiffSize - mae * mae);
        auto mre = relDiffSum1 / relDiffSize;
        auto rrmse = std::sqrt(relDiffSum2 / relDiffSize - mre * mre);

        std::cout << "Errors for " << label << std::endl;
        std::cout << std::scientific;
        std::cout << "    RMS error:  " << rmse << std::endl;
        std::cout << "    RRMS error: " << rrmse << std::endl;
        std::cout << "    MA error:   " << mae << std::endl;
        std::cout << "    MR error:   " << mre << std::endl;
        std::cout << std::fixed;
        std::cout << "    size:       " << absDiffSize << std::endl;
        std::cout << std::endl;
    }

    if (auto error = pricer.free(); !error.empty())
        return "benchPortfolio: " + error;

    return "";
}

kw::Error
    cmdBench(const Args& args)
{
    kw::Fd1dConfig config;
    config.theta = 0.5;
    config.tDim = args.at("-t").asLong();
    config.xDim = args.at("-x").asLong();
    config.xThreads = args.at("--cuda-x").asLong();
    config.nThreads = args.at("--cuda-n").asLong();

    auto batchCount = args.at("-n").asLong();
    auto batchSize = args.at("-b").asLong();

    double tolerance;
    if (auto error = kw::fromString(args.at("-e").asString(), tolerance); !error.empty())
        return "cmdBench: Fail to parse '-n <num>': " + error;

    if (auto error = printGpuInfo(); !error.empty())
        return "cmdBench: " + error;

    kw::Portfolio portfolio;
    if (auto error = kw::loadPortfolio(args.at("<portfolio>").asString(), portfolio); !error.empty())
        return "cmdBench: " + error;
    if (args.at("--amer").asBool() != args.at("--euro").asBool())
    {
        bool keepAmerican = args.at("--amer").asBool();
        bool keepEuropean = args.at("--euro").asBool();

        for (auto ii = portfolio.begin(); ii != portfolio.end(); )
        {
            const auto& asset = ii->first;
            if ((keepAmerican && !asset.e) || (keepEuropean && asset.e))
                portfolio.erase(ii++);
            else
                ++ii;
        }
    }
    if (args.at("--call").asBool() != args.at("--put").asBool())
    {
        bool keepCall = args.at("--call").asBool();
        bool keepPut = args.at("--put").asBool();

        for (auto ii = portfolio.begin(); ii != portfolio.end(); )
        {
            const auto& asset = ii->first;
            if ((keepCall && asset.w != kw::kParity::Call) || (keepPut && asset.w != kw::kParity::Put))
                portfolio.erase(ii++);
            else
                ++ii;
        }
    }
    //for (auto ii = portfolio.begin(); ii != portfolio.end(); )
    //{
    //    const auto& asset = ii->first;
    //    if (asset.r != asset.q)
    //        portfolio.erase(ii++);
    //    else
    //        ++ii;
    //}

    std::cout << "Portfolio" << std::endl;
    std::cout << "    Assets:      " << portfolio.size() << std::endl;
    std::cout << "    Batch count: " << batchCount << std::endl;
    std::cout << "    Batch size:  " << batchSize << std::endl;
    std::cout << std::endl;


    bool runAll = !args.at("--cpu32").asBool() && !args.at("--cpu64").asBool() &&
        !args.at("--gpu32").asBool() && !args.at("--gpu64").asBool();

    if (runAll || args.at("--cpu32").asBool())
    {
        const auto label = "Fd1d<float>::solve";
        if (auto error = benchPortfolio<float, kw::Fd1d<float>>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
        KW_BENCHMARK_PRINT(label);
    }

    if (runAll || args.at("--gpu32").asBool())
    {
        const auto label = "Fd1d_Gpu<float>::solve";
        if (auto error = benchPortfolio<float, kw::Fd1d_Gpu<float>>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
        KW_BENCHMARK_PRINT(label);
    }

    if (runAll || args.at("--cpu64").asBool())
    {
        const auto label = "Fd1d<double>::solve";
        if (auto error = benchPortfolio<double, kw::Fd1d<double>>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
        KW_BENCHMARK_PRINT(label);
    }

    if (runAll || args.at("--gpu64").asBool())
    {
        const auto label = "Fd1d_Gpu<double>::solve";
        if (auto error = benchPortfolio<double, kw::Fd1d_Gpu<double>>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
        KW_BENCHMARK_PRINT(label);
    }

    return "";
}



int main(int argc, char** argv)
{
    auto args = docopt::docopt(
        kw::usage,
        { argv + 1, argv + argc },
        true,
        "kwinto v0.1.0");

    if (args.at("-v").asBool())
    {
        std::cout << "Command-Line Arguments" << std::endl;
        for (auto const& arg : args) {
            std::cout << "    " << arg.first << ": " << arg.second << std::endl;
        }
        std::cout << std::endl;
    }


    kw::Error error;
    if (args["bench"])
        error = cmdBench(args);

    if (!error.empty())
    {
        std::cerr << error << std::endl;
        return 1;
    }

    return 0;
}

