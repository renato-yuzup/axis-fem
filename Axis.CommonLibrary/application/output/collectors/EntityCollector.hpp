#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis { 

namespace services { namespace messaging {
  class ResultMessage;
} } // namespace axis::services::messaging

namespace application { namespace output { namespace collectors {

/**
 * Represents any object that, upon request, navigates through the analysis numerical model
 * and extracts and forwards requested informations to the result workbook.
 */
class AXISCOMMONLIBRARY_API EntityCollector
{
public:
	virtual ~EntityCollector(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy(void) const = 0;

  /**
   * Queries if this collector is interest in a particular message to start result collection.
   *
   * @param message The message to check.
   *
   * @return true if it is of interest, false otherwise.
   */
  virtual bool IsOfInterest(const axis::services::messaging::ResultMessage& message) const = 0;

  /**
   * @brief Prepares this collector to capture data from numerical model in an analysis step.
   *        
   * @remark This method can be overriden; base implementation does nothing.
  **/
  virtual void Prepare(void);

  /**
   * @brief Tells the collector to free any resource used while capturing data (but not to destroy itself).
   *        
   * @remark This method can be overriden; base implementation does nothing.
  **/
  virtual void TearDown(void);

  /**
   * Returns a brief, concise and friendly description of data gathered by this collector.
   *
   * @return The description.
   */
  virtual axis::String GetFriendlyDescription(void) const = 0;
}; // EntityCollector

} } } } // namespace axis::application::output::collectors
