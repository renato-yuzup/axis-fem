#pragma once

#ifndef BOOST_ENABLE_ASSERT_HANDLER
#define BOOST_ENABLE_ASSERT_HANDLER
#endif

namespace boost
{
	void assertion_failed(char const * expr, char const * function, char const * file, long line);
}