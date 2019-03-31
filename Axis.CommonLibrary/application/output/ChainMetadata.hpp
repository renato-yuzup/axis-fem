#pragma once
#include "Foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis { namespace application { namespace output {

/**
 * Carries information about a result collection chain.
**/
class AXISCOMMONLIBRARY_API ChainMetadata
{
public:

  /**
   * Default constructor.
  **/
  ChainMetadata(void);

  /**
   * Constructor.
   *
   * @param title    Short name of the chain format.
   * @param fileName Chain output filename.
  **/
  ChainMetadata(const axis::String& title, const axis::String& fileName);

  /**
   * Constructor.
   *
   * @param title       Short name of the chain format.
   * @param fileName    Chain output filename.
   * @param description Format description.
  **/
  ChainMetadata(const axis::String& title, const axis::String& fileName, const axis::String& description);

  /**
   * Copy constructor.
   *
   * @param other The source object.
  **/
  ChainMetadata(const ChainMetadata& other);

  /**
   * Destructor.
  **/
  ~ChainMetadata(void);

  ChainMetadata& operator = (const ChainMetadata& other);

  /**
   * Returns the short name of the output format.
   *
   * @return The name.
  **/
  axis::String GetTitle(void) const;

  /**
   * Returns the output filename for the associated collection chain.
   *
   * @return The output file name.
  **/
  axis::String GetOutputFileName(void) const;

  /**
   * Returns a short description of the chain format.
   *
   * @return The short description.
  **/
  axis::String GetShortDescription(void) const;

  /**
   * Returns how many collectors are gathered in tis output chain.
   *
   * @return The collector count.
  **/
  int GetCollectorCount(void) const;

  /**
   * Returns a short descriptions of a collector in this chain.
   *
   * @param index Zero-based index of the collector.
   *
   * @return The collector description.
  **/
  axis::String operator [](int index) const;

  /**
   * Returns a short descriptions of a collector in this chain.
   *
   * @param index Zero-based index of the collector.
   *
   * @return The collector description.
  **/
  axis::String GetCollectorDescription(int index) const;

  /**
   * Returns if the associated chain is configured to append data to output.
   *
   * @return true if it is, false otherwise (that is, will overwrite data).
  **/
  bool WillAppendData(void) const;

  /**
   * Adds description to a collector in the associated output chain.
   *
   * @param description The collector description.
  **/
  void AddCollectorDescription(const axis::String& description);

  /**
   * Sets the short name of the collection chain format.
   *
   * @param title The title.
  **/
  void SetTitle(const axis::String& title) const;

  /**
   * Sets the output file name for the associated chain.
   *
   * @param filename Filename.
  **/
  void SetOutputFileName(const axis::String& filename) const;

  /**
   * Sets the short description for the format of the associated collection chain.
   *
   * @param description The description.
  **/
  void SetShortDescription(const axis::String& description) const;

  /**
   * Sets append information.
   *
   * @param state The append state.
  **/
  void SetAppendDataState(bool state);
private:
  class Pimpl;

  void Copy(const ChainMetadata& other);

  Pimpl *pimpl_;
};

} } } // namespace axis::application::output
