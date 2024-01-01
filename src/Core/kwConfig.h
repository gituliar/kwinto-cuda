#pragma once

#include "Core/kwCore.h"

#include <concepts>
#include <map>
#include <string>


namespace kw {

template <class T>
concept explicit_floating = std::is_floating_point_v<T>;

template <class T>
concept explicit_integral = std::is_integral_v<T>;

template <class T>
concept explicit_string = std::is_convertible_v<T, std::string_view>;


class Config {
private:
    //std::map<std::string, bool>
    //        m_bools;
    std::map<std::string, f64>
            m_doubles;
    std::map<std::string, i64>
            m_integers;
    std::map<std::string, std::string>
            m_strings;

public:
    f64 get(const std::string& key, explicit_floating auto defaultValue) const
    {
        const auto ii = m_doubles.find(key);
        return ii != m_doubles.end() ? ii->second : defaultValue;
    }

    i64 get(const std::string& key, explicit_integral auto defaultValue) const
    {
        const auto ii = m_integers.find(key);
        return ii != m_integers.end() ? ii->second : defaultValue;
    }

    std::string
        get(const std::string& key, const explicit_string auto& defaultValue) const
    {
        const auto ii = m_strings.find(key);
        return ii != m_strings.end() ? ii->second : defaultValue;
    }


    void erase(const std::string& key)
    {
        m_doubles.erase(key);
        m_integers.erase(key);
        m_strings.erase(key);
    }


    void set(const std::string& key, explicit_floating auto value)
    {
        m_doubles[key] = value;
    }
    void set(const std::string& key, explicit_integral auto value)
    {
        m_integers[key] = value;
    }
    void set(const std::string& key, const explicit_string auto& value)
    {
        m_strings[key] = value;
    }
};

}