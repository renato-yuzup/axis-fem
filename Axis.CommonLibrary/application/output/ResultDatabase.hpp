#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis { 

namespace foundation {
namespace uuids {
class Uuid;
} // namespace axis::foundation::uuids
namespace date_time {
class Timestamp;
} // namespace axis::foundation::date::time
} // namespace axis::foundation

namespace services { 
namespace messaging {
class ResultMessage;
} // namespace axis::services::messaging
namespace diagnostics { namespace information {
class SolverCapabilities;
} } // namespace axis::services::diagnostics::information
} // namespace axis::services

namespace domain { 
namespace algorithms { namespace messages {
class SnapshotStartMessage;
class SnapshotEndMessage;
} } // namespace axis::domain::algorithms::messages
namespace analyses {
class NumericalModel;
class AnalysisTimeline;
} } // namespace axis::domain::analyses

namespace application { 

namespace jobs {
class WorkFolder;
class AnalysisStepInformation;
} // namespace axis::application::jobs

namespace output {
  
namespace collectors { 
class NodeSetCollector;
class ElementSetCollector;
class GenericCollector;
class EntityCollector;
namespace messages {
class AnalysisStepStartMessage;  
} // namespace axis::application::output::collectors::messages
} // namespace axis::application::output::collectors
namespace workbooks {
class ResultWorkbook;
} // namespace axis::application::output::workbooks

/**
 * Manages simulation data collection in respect of a single data
 * output file or entity.
 */
class AXISCOMMONLIBRARY_API ResultDatabase
{
public:
  ResultDatabase(void);
  ~ResultDatabase(void);
  void Destroy(void) const;

  void OpenDatabase(axis::application::jobs::WorkFolder& workFolder);
  void CloseDatabase(void);
  bool IsOpen(void) const;
  bool HasCollectors(void) const;
  void StartStep(const axis::application::jobs::AnalysisStepInformation& stepInfo);
  void EndStep(void);
  void StartSnapshot(const axis::domain::algorithms::messages::SnapshotStartMessage& message);
  void EndSnapshot(const axis::domain::algorithms::messages::SnapshotEndMessage& message);

  axis::String GetFormatTitle(void) const;
  axis::String GetOutputFileName(void) const;
  axis::String GetFormatDescription(void) const;
  bool GetAppendState(void) const;
  int GetCollectorCount(void) const;
  const axis::application::output::collectors::EntityCollector& operator [](int index) const;
  const axis::application::output::collectors::EntityCollector& GetCollector(int index) const;

  void AddNodeCollector(axis::application::output::collectors::NodeSetCollector& collector);
  void AddElementCollector(axis::application::output::collectors::ElementSetCollector& collector);
  void AddGenericCollector(axis::application::output::collectors::GenericCollector& collector);
  void RemoveNodeCollector(axis::application::output::collectors::NodeSetCollector& collector);
  void RemoveElementCollector(axis::application::output::collectors::ElementSetCollector& collector);
  void RemoveGenericCollector(axis::application::output::collectors::GenericCollector& collector);

  void SetWorkbook(axis::application::output::workbooks::ResultWorkbook& workbook);

  void WriteResults(const axis::services::messaging::ResultMessage& message,
                   const axis::domain::analyses::NumericalModel& numericalModel);
private:
  class Pimpl;
  Pimpl *pimpl_;
}; // ResultDatabase

} } } // namespace axis::application::output
