#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/fwd/output_collectors.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "CollectorBuildResult.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

/**
 * Represents a factory capable to build collectors compatible to a given set of formats.
 */
class AXISPHYSALIS_API CollectorFactory
{
public:
	virtual ~CollectorFactory(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy(void) const = 0;

  /**
   * Checks if it is possible to build a collector from a given statement.
   *
   * @param formatName Name of the format to which the collector should be compatible.
   * @param begin      Iterator pointing to the beginning of the statement.
   * @param end        Iterator pointing to the end of the statement.
   *
   * @return Parsing operation result.
   */
	virtual axis::services::language::parsing::ParseResult TryParse(
                            const axis::String& formatName, 
                            const axis::services::language::iterators::InputIterator& begin, 
                            const axis::services::language::iterators::InputIterator& end) = 0;

  /**
   * Parse and build.
   *
   * @param formatName       Name of the format to which the collector should be compatible.
   * @param begin            Iterator pointing to the beginning of the statement.
   * @param end              Iterator pointing to the end of the statement.
   * @param model            Numerical model from which gather information.
   * @param [in,out] context The current parse context.
   *
   * @return Parsing and building operation result, containing the collector.
   */
	virtual CollectorBuildResult ParseAndBuild(
                            const axis::String& formatName, 
                            const axis::services::language::iterators::InputIterator& begin, 
                            const axis::services::language::iterators::InputIterator& end,
                            const axis::domain::analyses::NumericalModel& model, 
                            axis::application::parsing::core::ParseContext& context) = 0;
};

} } } } // namespace axis::application::factories::collectors
