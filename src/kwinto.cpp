#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

#include "docopt.h"

#define KW_BENCHMARK_ON
#include "kwBenchmark.h"

#include "Core/kwConfig.h"
#include "Math/kwFd1d.h"
#include "Core/kwTime.h"
#include "Pricer/kwPriceEngineFactory.h"
#include "kwPortfolio.h"
#include "kwVersion.h"


namespace kw {

constexpr char usage[] = R"(
kwinto - Analytics for Options Trading

Usage:
    kwinto bench [options] <portfolio>

Arguments:
    <portfolio>     Read options data from <portfolio> CSV file

Options:
    -b <num>        Price <num> options per batch [default: 128]
    --call          Include call options
    --cpu32         Run single-precision benchmark on CPU
    --cpu64         Run double-precision benchmark on CPU
    --cuda-n <num>  Run <num> threads per CUDA block over the n-dimension [default: 8]
    --cuda-x <num>  Run <num> threads per CUDA block over the x-dimension [default: 128]
    -e <num>        Reject option prices less than <num> from RRMS error stats [default: 0.5]
    --gpu32         Run single-precision benchmark on GPU
    --gpu64         Run double-precision benchmark on GPU
    -n <num>        Run <num> batches (when 0 run all) [default: 0]
    --put           Include put options
    --ql64          Run double-precision benchmark with QuantLib
    -t <num>        Use <num> points for t-grid [default: 512]
    -v              Show extra details, be verbose
    -x <num>        Use <num> points for x-grid [default: 512]

    -h --help       Print this screen
    --version       Print Kwinto version
)";

}

using Args = std::map<std::string, docopt::value>;


//kw::Error
//    printGpuInfo()
//{
//    int nDevices;
//    cudaGetDeviceCount(&nDevices);
//
//    for (int i = 0; i < nDevices; i++) {
//        cudaDeviceProp device;
//        cudaGetDeviceProperties(&device, i);
//
//        std::cout << "Devices Info #" << i <<  std::endl;
//        std::cout << "    Name:                 " << device.name << std::endl;
//        std::cout << "    Integrated:           " << device.integrated << std::endl;
//        std::cout << std::endl;
//
//        std::cout << "    Total SM:             " <<
//            device.multiProcessorCount << std::endl;
//        std::cout << "    Clock Rate:           " <<
//            device.clockRate / 1e3 << " MHz" << std::endl;
//        //std::cout << "    Warp size (threads):  " <<
//        //    device.warpSize << std::endl;
//        std::cout << "    32-bit Regs (per SM): " <<
//            device.regsPerMultiprocessor << std::endl;
//        std::cout << "    Max Blocks (per SM):  " <<
//            device.maxBlocksPerMultiProcessor << std::endl;
//        std::cout << "    Max Threads (per SM): " <<
//            device.maxThreadsPerMultiProcessor << std::endl;
//        std::cout << std::endl;
//
//        std::cout << "    Total Memory:         " <<
//            device.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
//        std::cout << "    Memory Clock Rate:    " <<
//            device.memoryClockRate / 1e3 << " MHz" << std::endl;
//        std::cout << "    Memory Bus Width:     " <<
//            device.memoryBusWidth << " bits" << std::endl;
//        std::cout << "    Peak Bandwidth:       " <<
//            2.0 * device.memoryClockRate * (device.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
//        std::cout << std::endl;
//    }
//
//    return "";
//}


template<typename Real>
kw::Error
    benchPortfolio(
        const kw::Portfolio& portfolio,
        kw::Config           config,
        size_t               batchSize,
        size_t               batchCount,
        double               tolerance,
        const std::string&   label)
{
    if (batchSize == 0)
        batchSize = portfolio.size();

    if (batchCount == 0)
        batchCount = (portfolio.size() + batchSize - 1) / batchSize;

    kw::sPtr<kw::Pricer> engine;
    if (auto error = kw::PriceEngineFactory::create(config, engine); !error.empty())
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
    double mae = 0; // Maximum Absolute Error
    kw::Option maeAsset;
    double mre = 0; // Maximum Relative Error
    kw::Option mreAsset;

    for (auto i = 0; i < batchCount; ++i) {
        const auto& assets = batches[i];

        // 1. Solve
        if (assets.size() == batchSize) {
            // Record only full-size batches
            KW_BENCHMARK_RESUME(label);
        }

        std::vector<double> prices;
        if (auto error = engine->price(assets, prices); !error.empty())
            return "benchPortfolio: " + error;

        if (assets.size() == batchSize) {
            KW_BENCHMARK_PAUSE(label);
        }

        // 2. Collect statistics
        for (auto j = 0; j < assets.size(); ++j) {
            const auto& asset = assets[j];
            const auto& price = portfolio.at(asset);

            double got = prices[j];

            if (price >= tolerance) {
                double absDiff = std::abs(price - got);
                double relDiff = absDiff / price;

                if (absDiff > mae) {
                    mae = absDiff;
                    maeAsset = asset;
                }
                if (relDiff > mre) {
                    mre = relDiff;
                    mreAsset = asset;
                }

                absDiffSum1 += absDiff;
                absDiffSum2 += absDiff * absDiff;
                absDiffSize++;

                relDiffSum1 += relDiff;
                relDiffSum2 += relDiff * relDiff;
                relDiffSize++;
            }
        }
    }

    KW_BENCHMARK_PRINT(label);

    {
        auto absDiffMean = absDiffSum1 / absDiffSize;
        auto rmse = std::sqrt(absDiffSum2 / absDiffSize - absDiffMean * absDiffMean);

        auto relDiffMean = relDiffSum1 / relDiffSize;
        auto rrmse = std::sqrt(relDiffSum2 / relDiffSize - relDiffMean * relDiffMean);

        std::cout << "Errors for " << label << std::endl;
        std::cout << std::scientific;
        std::cout << "       RMSE : " << rmse << std::endl;
        std::cout << "      RRMSE : " << rrmse << std::endl;
        std::cout << "        MAE : " << mae << std::endl;
        std::cout << "        MRE : " << mre << std::endl;
        std::cout << "  MAE Asset : " << maeAsset.asString() << std::endl;
        std::cout << "  MRE Asset : " << mreAsset.asString() << std::endl;
        std::cout << std::fixed;
        std::cout << "      total : " << absDiffSize << " options" << std::endl;
        std::cout << std::endl;
    }

    return "";
}

kw::Error
    cmdBench(const Args& args)
{
    kw::Config config;

    config.set("FD1D.THETA", 0.5);
    config.set("FD1D.T_GRID_SIZE", args.at("-t").asLong());
    config.set("FD1D.X_GRID_SIZE", args.at("-x").asLong());
    config.set("FD1D.X_THREADS", args.at("--cuda-x").asLong());
    config.set("FD1D.N_THREADS", args.at("--cuda-n").asLong());

    auto batchCount = args.at("-n").asLong();
    auto batchSize = args.at("-b").asLong();

    double tolerance;
    if (auto error = kw::fromString(args.at("-e").asString(), tolerance); !error.empty())
        return "cmdBench: Fail to parse '-n <num>': " + error;

    //if (auto error = printGpuInfo(); !error.empty())
    //    return "cmdBench: " + error;

    kw::Portfolio portfolio;
    if (auto error = kw::loadPortfolio(args.at("<portfolio>").asString(), portfolio); !error.empty())
        return "cmdBench: " + error;
    {
        bool keepAmerican = true; // args.at("--amer").asBool();
        bool keepEuropean = false; // args.at("--euro").asBool();

        for (auto ii = portfolio.begin(); ii != portfolio.end(); )
        {
            const auto& asset = ii->first;
            const auto& price = ii->second;
            if (keepAmerican && !asset.e || keepEuropean && asset.e || price < tolerance)
                portfolio.erase(ii++);
            else
                ++ii;
        }
    }
    if (args.at("--call").asBool() != args.at("--put").asBool()) {
        bool keepCall = args.at("--call").asBool();
        bool keepPut = args.at("--put").asBool();

        for (auto ii = portfolio.begin(); ii != portfolio.end(); )
        {
            const auto& asset = ii->first;
            if ((keepCall && asset.w != kw::Parity::Call) || (keepPut && asset.w != kw::Parity::Put))
                portfolio.erase(ii++);
            else
                ++ii;
        }
    }

    std::cout << "Portfolio" << std::endl;
    std::cout << "    Assets     : " << portfolio.size() << std::endl;
    std::cout << "    Batch count: " << batchCount << std::endl;
    std::cout << "    Batch size : " << batchSize << std::endl;
    std::cout << std::endl;

    bool runAll = !args.at("--cpu32").asBool() && !args.at("--cpu64").asBool() &&
        !args.at("--gpu32").asBool() && !args.at("--gpu64").asBool() && !args.at("--ql64").asBool();

    if (runAll || args.at("--cpu32").asBool())
    {
        config.set("PRICE_ENGINE.MODE", "FD1D_CPU32");

        const auto label = "Fd1d<float>::solve";
        if (auto error = benchPortfolio<float>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
    }

    if (runAll || args.at("--gpu32").asBool()) {
        config.set("PRICE_ENGINE.MODE", "FD1D_GPU32");

        const auto label = "Fd1d_Gpu<float>::solve";
        if (auto error = benchPortfolio<float>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
    }

    if (runAll || args.at("--cpu64").asBool()) {
        config.set("PRICE_ENGINE.MODE", "FD1D_CPU64");

        const auto label = "Fd1d<double>::solve";
        if (auto error = benchPortfolio<double>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
    }

    if (runAll || args.at("--gpu64").asBool()) {
        config.set("PRICE_ENGINE.MODE", "FD1D_GPU64");

        const auto label = "Fd1d_Gpu<double>::solve";
        if (auto error = benchPortfolio<double>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
    }

    if (runAll || args.at("--ql64").asBool()) {
        config.set("PRICE_ENGINE.MODE", "FD1D_QL");

        const auto label = "Fd1d_QuantLib::solve";
        if (auto error = benchPortfolio<double>(portfolio, config, batchSize, batchCount, tolerance, label); !error.empty())
            return "cmdBench: " + error;
    }

    return "";
}



int main(int argc, char** argv)
{
    auto version = "kwinto-cuda " + kw::Version::GIT_TAG + " (" + kw::Version::GIT_REV + ", " + kw::Version::BUILD_DATE + ")";


    auto args = docopt::docopt(
        kw::usage,
        { argv + 1, argv + argc },
        true,
        version);
    std::cout << version << '\n' << std::endl;

    if (args.at("-v").asBool()) {
        std::cout << "Command-Line Arguments" << std::endl;
        for (auto const& arg : args) {
            std::cout << "    " << arg.first << ": " << arg.second << std::endl;
        }
        std::cout << std::endl;
    }


    kw::Error error;
    if (args["bench"])
        error = cmdBench(args);

    if (!error.empty()) {
        std::cerr << error << std::endl;
        return 1;
    }

    return 0;
}

