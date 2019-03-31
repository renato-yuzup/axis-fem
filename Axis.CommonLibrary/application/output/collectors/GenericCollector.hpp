#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "EntityCollector.hpp"

namespace axis { 

namespace domain { namespace analyses {
  class NumericalModel;
} } // namespace axis::domain::analyses

namespace application { namespace output { 

namespace recordsets {
  class ResultRecordset;
} // namespace axis::application::output::recordsets

namespace collectors {


/**
 * Represents an entity collectors which may act on any 
 * entity of the numerical model.
 *
 * @sa EntityCollector
 */
class AXISCOMMONLIBRARY_API GenericCollector : public EntityCollector
{
public:
  virtual ~GenericCollector(void);

  virtual void Collect(const axis::services::messaging::ResultMessage& message, 
    axis::application::output::recordsets::ResultRecordset& recordset,
    const axis::domain::analyses::NumericalModel& numericalModel) = 0;
}; // GenericCollector

} } } } // namespace axis::application::output::collectors
