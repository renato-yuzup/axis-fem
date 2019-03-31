#pragma once
#include "application/jobs/AnalysisStep.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "System.hpp"
#include "services/diagnostics/Process.hpp"
#include "services/messaging/CollectorHub.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace axis { namespace application { namespace agents {

class AnalysisAgent : public axis::services::messaging::CollectorHub
{
private:
	axis::application::jobs::StructuralAnalysis *analysis_;
	axis::domain::analyses::NumericalModel *_model;
  axis::foundation::date_time::Timestamp _stepStartTime;
				
	axis::services::diagnostics::Process _analysisBeginState;
	axis::services::diagnostics::Process _analysisEndState;

	storage_space_type _analysisStartPhysMemSize;
	storage_space_type _analysisStartVirtualMemSize;

	long _errorCount;
	long _warningCount;
	long _stepErrorCount;
	long _stepWarningCount;
	axis::String _lastErrorMsg;

	AnalysisAgent(const AnalysisAgent& other);
	AnalysisAgent& operator =(const AnalysisAgent& other);

	axis::String BuildAnalysisTypeString( void ) const;

	void LogAnalysisInformation( void );
	void LogAnalysisSummary( void );
	void LogStepInformation( void );
	void LogAfterStepInformation( void );

	void ApplyCurrentStepBoundaryConditions(void);
	void ApplyBoundaryConditionSetToModel(axis::domain::collections::DofList& bcTypeList, 
                                        axis::domain::collections::DofList& generalList, 
                                        axis::domain::collections::BoundaryConditionCollection& bcList);
	void ClearBoundaryConditionsFromModel( void );
	void ClearBoundaryConditionSet( axis::domain::collections::DofList& dofList );
protected:
	virtual void ProcessEventMessageLocally( const axis::services::messaging::EventMessage& volatileMessage );
  virtual void ProcessResultMessageLocally(const axis::services::messaging::ResultMessage& volatileMessage);
public:
	AnalysisAgent(void);
	~AnalysisAgent(void);
	void SetUp(axis::application::jobs::StructuralAnalysis& analysis);

	void Run(void);
};		

} } } // namespace axis::application::agents
