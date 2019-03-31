#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/output/collectors/Direction3DState.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "application/fwd/output_collectors.hpp"
#include "application/fwd/parsing.hpp"
#include "CollectorBuildResult.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class NodeCollectorBuilder;

class AXISPHYSALIS_API GeneralNodeCollectorFactory
{
public:
  ~GeneralNodeCollectorFactory(void);

  void Destroy( void ) const;

  /**
   * Creates a new instance of this factory.
   *
   * @return A new instance.
   */
  static GeneralNodeCollectorFactory& Create(void);

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
  CollectorBuildResult ParseAndBuild(
    const axis::services::language::iterators::InputIterator& begin, 
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
   * @param builder          The builder which knows how to build each standard node collector.
   *
   * @return An object holding information about the parsing result and possibly the created collector.
   */
  CollectorBuildResult ParseAndBuild(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, 
    const axis::domain::analyses::NumericalModel& model, 
    axis::application::parsing::core::ParseContext& context, 
    NodeCollectorBuilder& builder );
private:
  class Pimpl;

  GeneralNodeCollectorFactory(void);
  GeneralNodeCollectorFactory(const GeneralNodeCollectorFactory&);
  GeneralNodeCollectorFactory& operator = (const GeneralNodeCollectorFactory&);

  Pimpl *pimpl_;
};

} } } } // namespace axis::application::factories::collectors
