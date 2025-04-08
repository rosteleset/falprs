#pragma once

#include <concepts>

#include <absl/strings/numbers.h>
#include <userver/formats/json.hpp>

template <typename T>
T convertToNumber(const userver::formats::json::Value& value, T default_value)
{
  if (value.IsMissing())
    return default_value;

  if (value.IsNull())
    return default_value;

  try
  {
    if (value.IsInt() || value.IsInt64() || value.IsUInt64() || value.IsDouble())
      return value.As<T>(default_value);

    if (value.IsString())
    {
      // check integral type
      if (std::is_integral_v<T>)
      {
        int64_t i;
        if (absl::SimpleAtoi(value.As<std::string>(), &i))
          return i;
        return default_value;
      }

      // check float
      if (std::is_same_v<T, float>)
      {
        float f;
        if (absl::SimpleAtof(value.As<std::string>(), &f))
          return f;
        return default_value;
      }

      // convert to double
      double d;
      if (absl::SimpleAtod(value.As<std::string>(), &d))
        return d;
      return default_value;
    }
  } catch (const std::exception&)
  {
  }

  return default_value;
}

inline bool convertToBool(const userver::formats::json::Value& value, const bool default_value)
{
  if (value.IsMissing())
    return default_value;

  if (value.IsNull())
    return default_value;

  if (value.IsBool())
    return value.As<bool>();

  if (value.IsInt() || value.IsInt64() || value.IsUInt64())
    return value.As<int64_t>() != 0;

  if (value.IsDouble())
    return value.As<double>() != 0.0;

  if (value.IsString())
  {
    bool b;
    if (absl::SimpleAtob(value.As<std::string>(), &b))
      return b;
    return default_value;
  }

  return default_value;
}

template <typename T>
  requires std::integral<T>
std::optional<T> convertToInt(const userver::formats::json::Value& value)
{
  if (value.IsMissing())
    return {};

  if (value.IsNull())
    return {};

  std::optional<T> result;
  try
  {
    T i;
    if (absl::SimpleAtoi(value.ConvertTo<std::string>(), &i))
      result = i;
    else
      result.reset();
  } catch (const std::exception&)
  {
    result.reset();
  }

  return result;
}

inline std::string convertToString(const userver::formats::json::Value& value)
{
  if (value.IsMissing())
    return {};

  if (value.IsNull())
    return {};

  if (value.IsString())
    return value.As<std::string>();

  if (value.IsInt() || value.IsInt64() || value.IsUInt64())
    return std::to_string(value.As<int64_t>());

  if (value.IsDouble())
    return std::to_string(value.As<double>());

  return {};
}
