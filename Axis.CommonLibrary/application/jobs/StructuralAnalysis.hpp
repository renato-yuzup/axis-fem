#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace axis { 

namespace domain {
namespace analyses {
class NumericalModel;
} // namespace analyses
} // namespace domain

namespace application { 
namespace jobs {

namespace monitoring {
class HealthMonitor;
} // namespace monitoring

class AnalysisStep;
class WorkFolder;

class AXISCOMMONLIBRARY_API StructuralAnalysis
{
public:
  StructuralAnalysis(const axis::String& workFolderPath);
  ~StructuralAnalysis(void);
  
  void Destroy(void) const;
  monitoring::HealthMonitor& GetHealthMonitor(void) const;
  axis::domain::analyses::NumericalModel& GetNumericalModel(void) const;
  void SetNumericalModel(axis::domain::analyses::NumericalModel& model);
  const axis::application::jobs::AnalysisStep& GetStep(int index) const;
  axis::application::jobs::AnalysisStep& GetStep(int index);
  void AddStep(axis::application::jobs::AnalysisStep& step);
  int GetStepCount(void) const;
  const WorkFolder& GetWorkFolder(void) const;
  WorkFolder& GetWorkFolder(void);

  axis::String GetTitle(void) const;
  void SetTitle(const axis::String& title);
  axis::foundation::uuids::Uuid GetId(void) const;
  void SetId(const axis::foundation::uuids::Uuid& newId);
  axis::foundation::date_time::Timestamp GetCreationDate(void) const;
  void SetCreationDate(const axis::foundation::date_time::Timestamp& newDate);
  int GetCurrentStepIndex(void) const;
  void SetCurrentStepIndex(int stepIndex);
  AnalysisStep& GetCurrentStep(void);
  const AnalysisStep& GetCurrentStep(void) const;
  axis::foundation::date_time::Timestamp GetAnalysisStartTime(void) const;
  axis::foundation::date_time::Timestamp GetAnalysisEndTime(void) const;
  void SetAnalysisStartTime(const axis::foundation::date_time::Timestamp& startTime);
  void SetAnalysisEndTime(const axis::foundation::date_time::Timestamp& endTime);
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} //namespace jobs
} // namespace application
} // namespace axis
