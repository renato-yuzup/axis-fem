#include "NodalVelocityConstraintFactory.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "domain/boundary_conditions/VariableConstraint.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace aafc = axis::application::factories::boundary_conditions;
namespace aaj = axis::application::jobs;
namespace adcv = axis::domain::curves;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;

aafc::NodalVelocityConstraintFactory::NodalVelocityConstraintFactory( void ) :
VariableConstraintFactory(_T("VELOCITY"))
{
	// nothing to do here
}

void aafc::NodalVelocityConstraintFactory::RegisterNewBoundaryCondition( aaj::AnalysisStep& step, 
  ade::DoF& dof, afm::RelativePointer& curvePtr, real scaleFactor, real releaseTime )
{
	adbc::BoundaryCondition& bc = *new adbc::VariableConstraint(
    adbc::BoundaryCondition::PrescribedVelocity, curvePtr, scaleFactor, releaseTime);
	step.Velocities().Add(dof, bc);
}
