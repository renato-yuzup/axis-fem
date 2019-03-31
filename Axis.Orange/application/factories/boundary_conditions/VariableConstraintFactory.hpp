#pragma once
#include "application/factories/boundary_conditions/ConstraintFactory.hpp"
#include "AxisString.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"

namespace axis { namespace application { namespace factories { namespace boundary_conditions {

class VariableConstraintFactory : 
  public axis::application::factories::boundary_conditions::ConstraintFactory
{
public:
  VariableConstraintFactory(const axis::String& constraintStatementName);
  ~VariableConstraintFactory(void);

  virtual axis::services::language::parsing::ParseResult TryParse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
  virtual axis::services::language::parsing::ParseResult ParseAndBuild( 
    axis::application::jobs::StructuralAnalysis& analysis, 
    axis::application::parsing::core::ParseContext& context, 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
protected:
	virtual void RegisterNewBoundaryCondition(axis::application::jobs::AnalysisStep& step, 
    axis::domain::elements::DoF& dof, axis::foundation::memory::RelativePointer& curvePtr, 
    real scaleFactor, real releaseTime) = 0;
private:
	class ParseParameters
	{
	public:
		ParseParameters(void);

		axis::String NodeSetId;
		axis::String CurveId;
		bool Directions[3];
		real ScalingFactor;
    real ReleaseTime;
		axis::services::language::parsing::ParseResult ParsingResult;
	};

	void InitGrammar(const axis::String& constraintStatementName);
	ParseParameters ParseVariableConstraintStatement(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end) const;
	bool IsNodeSetAvailable( const axis::String& nodeSetId, 
    axis::domain::analyses::NumericalModel& analysis, 
    axis::application::parsing::core::ParseContext& context ) const;
	bool CurveExists( const axis::String& curveId, 
    const axis::domain::analyses::NumericalModel& model, 
    axis::application::parsing::core::ParseContext& context ) const;
	bool AreDofsAvailable( const axis::String& nodeSetId, bool *enabledDirections,
    axis::application::jobs::StructuralAnalysis& analysis, 
    axis::application::parsing::core::ParseContext& context ) const;
	void BuildVariableConstraint( const axis::String& nodeSetId, 
    const axis::String& curveId, bool *enabledDirections, real scaleFactor, 
    real releaseTime,
    axis::application::jobs::StructuralAnalysis& analysis, 
    axis::application::parsing::core::ParseContext& context );
	axis::String GetBoundaryConditionSymbolName(unsigned long nodeId, int dofIndex, 
    int stepIndex) const;

	axis::services::language::primitives::GeneralExpressionParser _variableConstraintExpression;
	axis::services::language::primitives::GeneralExpressionParser _shortVariableConstraintExpression;
	axis::services::language::primitives::OrExpressionParser _directionReservedWord;
	axis::services::language::primitives::OrExpressionParser _idExpression;
	axis::services::language::primitives::EnumerationParser *_dofExpression;
	axis::services::language::primitives::OrExpressionParser _dofOperators;
  axis::services::language::primitives::GeneralExpressionParser _scaleExpression;
  axis::services::language::primitives::GeneralExpressionParser _releaseExpression;
  axis::services::language::primitives::GeneralExpressionParser _optExpr1;
  axis::services::language::primitives::GeneralExpressionParser _optExpr2;
  axis::services::language::primitives::GeneralExpressionParser _singleOptExpr1;
  axis::services::language::primitives::GeneralExpressionParser _singleOptExpr2;
  axis::services::language::primitives::OrExpressionParser _optionalExpressions;
};

} } } } // namespace axis::application::factories::boundary_conditions

