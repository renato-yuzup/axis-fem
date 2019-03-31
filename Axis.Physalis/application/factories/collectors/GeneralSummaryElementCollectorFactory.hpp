#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/output/collectors/summarizers/SummaryType.hpp"
#include "application/fwd/output_collectors.hpp"
#include "application/fwd/parsing.hpp"
#include "SummaryElementCollectorBuilder.hpp"
#include "CollectorBuildResult.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class AXISPHYSALIS_API GeneralSummaryElementCollectorFactory
{
public:
  ~GeneralSummaryElementCollectorFactory(void);
  void Destroy( void ) const;

  /**
   * Creates a new instance of this factory.
   *
   * @return A new instance.
   */
  static GeneralSummaryElementCollectorFactory& Create(void);

  /**
   * Tries to parse and resolve a statement for any known collectors of this factory.
   *
   * @param begin      The iterator pointing to the beginning of the statement.
   * @param end        The iterator pointing to the end of the statement.
   *
   * @return The best match result found for this statement.
   */
  axis::services::language::parsing::ParseResult TryParse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );

  /**
   * Parses a statement and build the corresponding collector, which is one of those
   * known by this factory.
   *
   * @param begin            The iterator pointing to the beginning of the statement.
   * @param end              The iterator pointing to the end of the statement.
   * @param model            The analysis numerical model.
   * @param [in,out] context The parsing context.
   *
   * @return An object holding information about the parsing result and possibly the created collector.
   */
  CollectorBuildResult ParseAndBuild(const axis::services::language::iterators::InputIterator& begin, 
                                     const axis::services::language::iterators::InputIterator& end, 
                                     const axis::domain::analyses::NumericalModel& model, 
                                     axis::application::parsing::core::ParseContext& context );

  /**
   * Parses a statement and build the corresponding collector, which is one of those
   * known by this factory.
   *
   * @param begin            The iterator pointing to the beginning of the statement.
   * @param end              The iterator pointing to the end of the statement.
   * @param model            The analysis numerical model.
   * @param [in,out] context The parsing context.
   * @param builder          Object to which will be delegated the construction process of the collectors.
   *
   * @return An object holding information about the parsing result and possibly the created collector.
   */
  CollectorBuildResult ParseAndBuild(const axis::services::language::iterators::InputIterator& begin, 
                                     const axis::services::language::iterators::InputIterator& end, 
                                     const axis::domain::analyses::NumericalModel& model, 
                                     axis::application::parsing::core::ParseContext& context,
                                     SummaryElementCollectorBuilder& builder);
private:
  class Pimpl;

  GeneralSummaryElementCollectorFactory(void);
  GeneralSummaryElementCollectorFactory(const GeneralSummaryElementCollectorFactory&);
  GeneralSummaryElementCollectorFactory& operator = (const GeneralSummaryElementCollectorFactory&);

  Pimpl *pimpl_;
};

} } } } // namespace axis::application::factories::collectors
