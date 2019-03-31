#pragma once
#include "foundation/Axis.Capsicum.hpp"

namespace axis { namespace application { namespace scheduling { namespace broker {

/**
 * Indicates why an analysis cannot be run using current hardware and
 * configuration.
 */
enum class AXISCAPSICUM_API ClashReason
{
  ///< Current hardware and configuration is sufficient for running analysis.
  kSuccessful,
  ///< Requested analysis running mode cannot be fulfilled because no
  // compatible hardware were found.
  kNoResourcesAvailable,
  ///< Analysis cannot be run because it was not possible to seek a common
  // running mode for all elements in the analysis.
  kCapabilityClash,
  ///< Requested analysis running mode cannot be fulfilled because current
  // configuration does not allow.
  kDeniedByEnvironment
};

} } } } // namespace axis::application::scheduling::broker
