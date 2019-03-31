#pragma once
#include "foundation/Axis.Core.hpp"
#include "services/io/StreamWriter.hpp"
#include "services/messaging/MessageListener.hpp"
#include "foundation/AxisException.hpp"
#include "services/locales/Locale.hpp"
#include "services/messaging/WarningMessage.hpp"
#include "services/messaging/InfoMessage.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "services/messaging/LogMessage.hpp"

namespace axis { namespace services { namespace logging {

class AXISCORE_API LogFile : public axis::services::messaging::MessageListener
{
public:
  virtual ~LogFile(void);
  void Destroy(void) const;
  void StartLogging(void);
  bool IsReady(void) const;
  void StopLogging(void);
  void PrintLine(const axis::String& line);

	static LogFile& Create(const axis::String& filename, bool appendMode);
	static LogFile& Create(const axis::String& filename);
	static LogFile& Create(axis::services::io::StreamWriter& writer);
protected:
	virtual void DoProcessEventMessage( axis::services::messaging::EventMessage& volatileMessage );
	void LogInfoMessage( axis::services::messaging::InfoMessage& message ) const;
	void LogWarningMessage( axis::services::messaging::WarningMessage& message ) const;
	void LogErrorMessage( axis::services::messaging::ErrorMessage& message ) const;
	void WriteLogMessage( axis::services::messaging::LogMessage& message );
private:
  LogFile(const axis::String& filename, bool appendMode);
  LogFile(const axis::String& filename);
  LogFile(axis::services::io::StreamWriter& writer);
  axis::services::io::StreamWriter *OpenFileStream(const axis::String& filename, 
    bool appendMode = true) const;
  void PrintHeader(const axis::String& heading) const;
  void PrintHeader(const axis::String& heading, const axis::String::char_type pattern) const;
  void PrintException(const axis::foundation::AxisException& e) const;
  axis::String BuildSourceInformationString(const axis::services::messaging::Message& message) const;

  axis::services::io::StreamWriter *_writer;
  axis::services::io::StreamWriter::WriteMode _writeMode;
  bool _onRootSection;
  bool _isInsideSection;
  int _nestingIndex;
  axis::String _padding;
  const axis::services::locales::Locale& _currentLocale;
};

} } } // namespace axis::services::logging
