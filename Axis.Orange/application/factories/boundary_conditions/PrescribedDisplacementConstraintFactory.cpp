#include "PrescribedDisplacementConstraintFactory.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "domain/boundary_conditions/VariableConstraint.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace aafbc = axis::application::factories::boundary_conditions;
namespace aaj = axis::application::jobs;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;

aafbc::PrescribedDisplacementConstraintFactory::PrescribedDisplacementConstraintFactory( void ) :
VariableConstraintFactory(_T("DISPLACEMENT"))
{
	// nothing to do here
}

void aafbc::PrescribedDisplacementConstraintFactory::RegisterNewBoundaryCondition( 
aaj::AnalysisStep& step, ade::DoF& dof, afm::RelativePointer& curvePtr, real scaleFactor, real releaseTime )
{
	adbc::BoundaryCondition& bc = 
    *new adbc::VariableConstraint(adbc::BoundaryCondition::PrescribedDisplacement, 
    curvePtr, scaleFactor, releaseTime);
	step.Displacements().Add(dof, bc);
}
