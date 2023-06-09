#pragma once

#include "kwFd1dCpu.h"
#include "kwFd1dGpu.h"
#include "kwFd1dQl.h"

#include "Math/kwFd1d.h"
#include "Utils/kwConfig.h"


namespace kw
{

class PriceEngineFactory
{
public:
    static Error
        create(const Config& config, sPtr<PriceEngine>& engine)
    {
        std::string mode = config.get("PRICE_ENGINE.MODE", "");
        if (mode.empty())
            return "PriceEngineFactory: Missing PRICE_ENGINE.MODE key";

        if (mode == "FD1D_CPU32")
        {
            engine = make_sPtr<Fd1dCpu_PriceEngine<float>>();
            if (auto error = engine->init(config); !error.empty())
                return "PriceEngineFactory: " + error;
            return "";
        }

        if (mode == "FD1D_CPU64")
        {
            engine = make_sPtr<Fd1dCpu_PriceEngine<double>>();
            if (auto error = engine->init(config); !error.empty())
                return "PriceEngineFactory: " + error;
            return "";
        }

        if (mode == "FD1D_GPU32")
        {
            engine = make_sPtr<Fd1dGpu_PriceEngine<float>>();
            if (auto error = engine->init(config); !error.empty())
                return "PriceEngineFactory: " + error;
            return "";
        }

        if (mode == "FD1D_GPU64")
        {
            engine = make_sPtr<Fd1dGpu_PriceEngine<double>>();
            if (auto error = engine->init(config); !error.empty())
                return "PriceEngineFactory: " + error;
            return "";
        }

        if (mode == "FD1D_QL")
        {
            engine = make_sPtr<Fd1dQl_PriceEngine>();
            if (auto error = engine->init(config); !error.empty())
                return "PriceEngineFactory: " + error;
            return "";
        }

        return "PriceEngineFactory: Unknown PRICE_ENGINE.MODE '" + mode + "'";
    }
};

}