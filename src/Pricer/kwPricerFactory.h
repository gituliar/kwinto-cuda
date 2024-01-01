#pragma once

#include "Core/kwConfig.h"
#include "Pricer/kwBlackScholes.h"
#include "Pricer/kwFd1d.h"


namespace kw
{

class PricerFactory
{
public:
    static Error
        create(const Config& config, sPtr<Pricer>& pricer)
    {
        std::string mode = config.get("PRICER", "");
        if (mode.empty())
            return "PricerFactory: Missing PRICER key";

        if (mode == "BS")
        {
            pricer = make_sPtr<BlackScholes_Pricer>();
            if (auto error = pricer->init(config); !error.empty())
                return "PricerFactory: " + error;
            return "";
        }

        if (mode == "FD1D")
        {
            pricer = make_sPtr<Fd1d_Pricer>();
            if (auto error = pricer->init(config); !error.empty())
                return "PricerFactory: " + error;
            return "";
        }

        return "PricerFactory: Unknown PRICER = " + mode;
    }
};

}