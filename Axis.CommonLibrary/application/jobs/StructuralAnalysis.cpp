#include "StructuralAnalysis.hpp"
#include "StructuralAnalysis_Pimpl.hpp"
#include "WorkFolder.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/OutOfBoundsException.hpp"

using axis::application::jobs::AnalysisStep;
using axis::application::jobs::WorkFolder;
using axis::application::jobs::StructuralAnalysis;
using axis::application::jobs::monitoring::HealthMonitor;
using axis::domain::analyses::NumericalModel;
using axis::foundation::date_time::Timestamp;
using axis::foundation::uuids::Uuid;

StructuralAnalysis::StructuralAnalysis(const axis::String& workFolderPath)
{
  pimpl_ = new Pimpl();
  pimpl_->workFolder = new WorkFolder(workFolderPath);
}

StructuralAnalysis::~StructuralAnalysis(void)
{
  delete pimpl_->workFolder;
  delete pimpl_;
  pimpl_ = NULL;
}

void StructuralAnalysis::Destroy( void ) const
{
  delete this;
}

HealthMonitor& StructuralAnalysis::GetHealthMonitor( void ) const
{
  return pimpl_->monitor;
}

NumericalModel& StructuralAnalysis::GetNumericalModel( void ) const
{
  if (pimpl_->model == NULL)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return *pimpl_->model;
}

void StructuralAnalysis::SetNumericalModel( NumericalModel& model )
{
  pimpl_->model = &model;
}

const AnalysisStep& StructuralAnalysis::GetStep( int index ) const
{
  if (index < 0 || index >= pimpl_->steps.size())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return *pimpl_->steps[index];
}

AnalysisStep& StructuralAnalysis::GetStep( int index )
{
  if (index < 0 || index >= pimpl_->steps.size())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return *pimpl_->steps[index];
}

void StructuralAnalysis::AddStep( AnalysisStep& step )
{
  pimpl_->steps.push_back(&step);
}

int StructuralAnalysis::GetStepCount( void ) const
{
  return (int)pimpl_->steps.size();
}

const WorkFolder& StructuralAnalysis::GetWorkFolder( void ) const
{
  return *pimpl_->workFolder;
}

WorkFolder& StructuralAnalysis::GetWorkFolder( void )
{
  return *pimpl_->workFolder;
}

axis::String StructuralAnalysis::GetTitle( void ) const
{
  return pimpl_->title;
}

void StructuralAnalysis::SetTitle( const axis::String& title )
{
  pimpl_->title = title;
}

Uuid StructuralAnalysis::GetId( void ) const
{
  return pimpl_->jobId;
}

void StructuralAnalysis::SetId( const Uuid& newId )
{
  pimpl_->jobId = newId;
}

Timestamp StructuralAnalysis::GetCreationDate( void ) const
{
  return pimpl_->creationDate;
}

void StructuralAnalysis::SetCreationDate( const Timestamp& newDate )
{
  pimpl_->creationDate = newDate;
}

int StructuralAnalysis::GetCurrentStepIndex( void ) const
{
  return pimpl_->currentStepIndex;
}

void StructuralAnalysis::SetCurrentStepIndex( int stepIndex )
{
  if (stepIndex < -1 || stepIndex >= (int)pimpl_->steps.size())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  pimpl_->currentStepIndex = stepIndex;
}

AnalysisStep& StructuralAnalysis::GetCurrentStep( void )
{
  if (pimpl_->currentStepIndex == -1)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return *pimpl_->steps[pimpl_->currentStepIndex];
}

const AnalysisStep& StructuralAnalysis::GetCurrentStep( void ) const
{
  if (pimpl_->currentStepIndex == -1)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return *pimpl_->steps[pimpl_->currentStepIndex];
}

Timestamp StructuralAnalysis::GetAnalysisStartTime( void ) const
{
  return pimpl_->analysisStartTime;
}

Timestamp StructuralAnalysis::GetAnalysisEndTime( void ) const
{
  return pimpl_->analysisEndTime;
}

void StructuralAnalysis::SetAnalysisStartTime( const Timestamp& startTime )
{
  pimpl_->analysisStartTime = startTime;
}

void StructuralAnalysis::SetAnalysisEndTime( const Timestamp& endTime )
{
  pimpl_->analysisEndTime = endTime;
}
