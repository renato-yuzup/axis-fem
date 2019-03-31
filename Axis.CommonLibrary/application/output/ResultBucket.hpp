#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { 

namespace services { namespace messaging {
class ResultMessage;
} } // namespace axis::services::messaging

namespace domain { namespace analyses {
class NumericalModel;
} } // namespace axis::domain::analyses

namespace application { namespace output {

class ResultDatabase;
class ChainMetadata;

/**********************************************************************************************//**
 * @class ResultBucket
 *
 * @brief Manages result recording operations for a given analysis step.
 **************************************************************************************************/
class AXISCOMMONLIBRARY_API ResultBucket
{
public:
  ResultBucket(void);
  virtual ~ResultBucket(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy(void) const = 0;

  /**
   * Process the result.
   *
   * @param message        The message.
   * @param numericalModel The numerical model.
   */
  virtual void PlaceResult(const axis::services::messaging::ResultMessage& message, 
                           const axis::domain::analyses::NumericalModel& numericalModel) = 0;

  /**
   * Returns information about a chain stored in this bucket.
   *
   * @param index Zero-based index of the chain.
   *
   * @return The chain metadata.
   */
  virtual ChainMetadata GetChainMetadata(int index) const = 0;

  /**
   * Returns how many chains are stored in this bucket.
   *
   * @return The chain count.
   */
  virtual int GetChainCount(void) const = 0;
};

} } } // namespace axis::application::output
