#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/output/collectors/Direction3DState.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "application/fwd/output_collectors.hpp"
#include "application/fwd/parsing.hpp"
#include "CollectorBuildResult.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class ElementCollectorBuilder;

/**
 * Builds general purpose collectors which acts on elements.
 */
class AXISPHYSALIS_API GeneralElementCollectorFactory
{
public:
  /**
   * Destructor.
   */
  ~GeneralElementCollectorFactory(void);

  /**
   * Destroys this object.
   */
  void Destroy( void ) const;

  /**
   * Creates a new instance of this factory.
   *
   * @return The new instance.
   */
  static GeneralElementCollectorFactory& Create(void);

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
  CollectorBuildResult ParseAndBuild( const axis::services::language::iterators::InputIterator& begin, 
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
   * @param builder          Builder to which delegate the construction process of each collector.
   *
   * @return An object holding information about the parsing result and possibly the created collector.
   */
  CollectorBuildResult ParseAndBuild( const axis::services::language::iterators::InputIterator& begin, 
                                      const axis::services::language::iterators::InputIterator& end, 
                                      const axis::domain::analyses::NumericalModel& model, 
                                      axis::application::parsing::core::ParseContext& context,
                                      ElementCollectorBuilder& builder);
private:
  class Pimpl;

  GeneralElementCollectorFactory(void);
  GeneralElementCollectorFactory(const GeneralElementCollectorFactory&);
  GeneralElementCollectorFactory& operator = (const GeneralElementCollectorFactory&);

  Pimpl *pimpl_;
};

} } } } // namespace axis::application::factories::collectors
