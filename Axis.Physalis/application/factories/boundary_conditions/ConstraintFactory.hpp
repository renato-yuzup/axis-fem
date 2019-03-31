#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/jobs/StructuralAnalysis.hpp"

namespace axis { namespace application { namespace factories { namespace boundary_conditions {

class AXISPHYSALIS_API ConstraintFactory
{
public:
	virtual ~ConstraintFactory(void);

	virtual axis::services::language::parsing::ParseResult TryParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end) = 0;
	virtual axis::services::language::parsing::ParseResult ParseAndBuild(
    axis::application::jobs::StructuralAnalysis& analysis, 
    axis::application::parsing::core::ParseContext& context, 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end) = 0;
};

} } } } // namespace axis::application::factories::boundary_conditions
