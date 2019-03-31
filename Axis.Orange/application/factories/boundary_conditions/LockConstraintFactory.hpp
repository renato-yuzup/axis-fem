#pragma once
#include "application/factories/boundary_conditions/ConstraintFactory.hpp"
#include "AxisString.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"

namespace axis { namespace application { namespace factories { namespace boundary_conditions {

class LockConstraintFactory : public axis::application::factories::boundary_conditions::ConstraintFactory
{
public:
	LockConstraintFactory(void);
	~LockConstraintFactory(void);
	virtual axis::services::language::parsing::ParseResult TryParse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
	virtual axis::services::language::parsing::ParseResult ParseAndBuild( 
    axis::application::jobs::StructuralAnalysis& analysis, 
    axis::application::parsing::core::ParseContext& context, 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
private:
	class ParseParameters
	{
	public:
		axis::String NodeSetId;
		bool Directions[3];
		axis::services::language::parsing::ParseResult ParsingResult;
    real ReleaseTime;

		ParseParameters(void);
	};

  void InitGrammar(void);
  ParseParameters ParseConstraint(const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end) const;
  bool NodeSetExists( const axis::String& nodeSetId, 
    axis::domain::analyses::NumericalModel& analysis, 
    axis::application::parsing::core::ParseContext& context ) const;
  bool CheckForRedefinition( const axis::String& nodeSetId, bool *enabledDirections, 
    axis::application::jobs::StructuralAnalysis& analysis, 
    axis::application::parsing::core::ParseContext& context ) const;
  void BuildClampingConstraints( const axis::String& nodeSetId, 
    bool *enabledDirections, real releaseTime,
    axis::application::jobs::StructuralAnalysis& analysis, 
    axis::application::parsing::core::ParseContext& context ) const;
  axis::String GetBoundaryConditionSymbolName(unsigned long nodeId, 
    int dofIndex, int stepIndex) const;

	axis::services::language::primitives::GeneralExpressionParser _constraintExpression;
	axis::services::language::primitives::OrExpressionParser _directionReservedWord;
	axis::services::language::primitives::OrExpressionParser _nodeSetIdExpression;
	axis::services::language::primitives::EnumerationParser *_dofExpression;
	axis::services::language::primitives::OrExpressionParser _dofOperators;
  axis::services::language::primitives::GeneralExpressionParser _releaseExpr;
  axis::services::language::primitives::OrExpressionParser _optionalExpr;
};

} } } } // namespace axis::application::factories::boundary_conditions
