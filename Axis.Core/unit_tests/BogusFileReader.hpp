#pragma once

#include "services/io/FileReader.hpp"

class BogusFileReader : public axis::services::io::FileReader
{
private:
	int _lineCountToFail;
public:
	BogusFileReader(const axis::String& filename);
	~BogusFileReader(void);

	virtual void ReadLine( axis::String& s );
};

