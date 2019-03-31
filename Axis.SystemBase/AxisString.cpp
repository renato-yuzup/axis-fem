#pragma once
#include "stdafx.h"
#include "AxisString.hpp"
#include <string>
#include <sstream>
#include <tchar.h>
#include <xstring>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/functional/hash/extensions.hpp>
#include "System.hpp"
#include "foundation/memory/HeapStackArena.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"

#define AXISSTRING_MIN_BUF_LEN		    10
#define AXISSTRING_MAXSIZE			      4294967291
#define AXISSTRING_MAX_BUFF_RESERVE	  4096
#define AXISSTRING_BUFF_GROWTH		    1.15
#define AXISSTRING_BUFF_THRESHOLD	    4

namespace {
	size_t inline calculate_buffer_size(size_t strLen)
	{
		return ((strLen*(AXISSTRING_BUFF_GROWTH) > AXISSTRING_MAX_BUFF_RESERVE?	
						(strLen + AXISSTRING_MAX_BUFF_RESERVE) :					
						(strLen >= AXISSTRING_BUFF_THRESHOLD?					
								(size_t)(strLen * AXISSTRING_BUFF_GROWTH) :		
								strLen + AXISSTRING_MIN_BUF_LEN)) + 1);
	}

	bool inline is_buffer_too_big(size_t strLen, size_t buffSize)
	{
		return (buffSize - strLen - 1 > AXISSTRING_MAX_BUFF_RESERVE);
	}
	bool inline is_buffer_too_small(size_t strLen, size_t buffSize)
	{
		return (strLen + 1 > buffSize);
	}
	bool inline needs_reallocate_buffer(size_t strLen, size_t buffSize)
	{
		return is_buffer_too_small(strLen, buffSize) || is_buffer_too_big(strLen, buffSize);
	}
}

template<class T>
const size_t axis::AxisString<T>::npos = -1;

template<class T>
const unsigned int axis::AxisString<T>::fp_parse_precision = 16;

template<class T>
axis::AxisString<T>::AxisString( void )
{
	_buf = NULL;
	init_buffer();
}
template<class T>
axis::AxisString<T>::AxisString( const self& str )
{
	_buf = NULL; 
  _size = 0;
	copy(str);
}

template<class T>
axis::AxisString<T>::AxisString( const T *s, size_t n )
{
	_buf = NULL;
  _size = 0;
	copy(s, n);
}

template<class T>
axis::AxisString<T>::AxisString( size_t n, T c )
{
	_buf = NULL;
  _size = 0;
	copy(&c, 1, n-1);
	for (size_t i = 1; i < n; ++i)
	{
		_buf[i] = c;
	}
	_buf[n] = NULL;	// put null terminator
	_str_len = n;
}

template<class T>
axis::AxisString<T>::AxisString( T c )
{
	_buf = NULL;
  _size = 0;
	copy(&c, 1);
}

template<class T>
axis::AxisString<T>::AxisString( const self& str, size_t pos, size_t n /*= npos*/ )
{
	_buf = NULL;
  _size = 0;
	const char_type *buffer = str.data();
	buffer += pos;
	copy(buffer, n);
}

template<class T>
axis::AxisString<T>::AxisString( const T *s )
{
	_buf = NULL;
  _size = 0;
	copy(s);
}

template<class T>
axis::AxisString<T>::AxisString( const_iterator& begin, const_iterator& end )
{
	_buf = NULL;
  _size = 0;
	copy(begin.to_string(end));
}

template<class T>
axis::AxisString<T>::AxisString( const_reverse_iterator& begin, const_reverse_iterator& end )
{
	_buf = NULL;
  _size = 0;
	copy(begin.to_string(end));
}

template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::operator=( const self& s )
{
	if (&s == this) return *this;
	copy(s);
	return *this;
}

template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::operator=( const T *s )
{
	if (s == _buf) return *this;
	copy(s);
	return *this;
}

template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::operator=( const T c )
{
	copy(&c, 1);
	return *this;
}

template<class T>
void axis::AxisString<T>::init_buffer(void)
{
#if defined AXIS_NO_MEMORY_ARENA
	_buf = new T[AXISSTRING_MIN_BUF_LEN];
#else
  _buf = (char_type *)System::StringMemory().Allocate(AXISSTRING_MIN_BUF_LEN*sizeof(char_type));
#endif
	_buf[0] = NULL;
	_size = AXISSTRING_MIN_BUF_LEN;
	_str_len = 0;
}

template<class T>
void axis::AxisString<T>::free_buffer(void)
{
	if (_buf != NULL)
	{
#if defined AXIS_NO_MEMORY_ARENA
		delete [] _buf;
#else
    System::StringMemory().Deallocate(_buf);
#endif
	}
	_buf = NULL;
	_size = 0;
	_str_len = 0;
}

template<class T>
void axis::AxisString<T>::copy(const self& s)
{
	if (&s == this)
	{	// self assignment, ignore
		return;
	}

	T *writeBuf = _buf;
	size_t strLen = s.size();
	size_t buffLen = _size;

	// check if we need to re-allocate space
	if (needs_reallocate_buffer(strLen, buffLen))
	{
		buffLen = calculate_buffer_size(strLen);
#if defined AXIS_NO_MEMORY_ARENA
		writeBuf = new T[buffLen];
#else
    writeBuf = (char_type *)System::StringMemory().Allocate(buffLen*sizeof(char_type));
#endif
	}

	try
	{
		const T *srcBuf = s.data();
		for (size_t i = 0; i < strLen; i++)
		{
			writeBuf[i] = srcBuf[i];
		}
		writeBuf[strLen] = NULL;
	}
	catch (...)
	{	// if something wrong happens, we must delete buffer from the free store
    if (writeBuf != _buf) 
    {
#if defined AXIS_NO_MEMORY_ARENA
      delete [] writeBuf;
#else
      System::StringMemory().Deallocate(writeBuf);
#endif
    }
		throw;
	}

	// if we re-allocated earlier, free old buffer
	if (writeBuf != _buf)
	{
		free_buffer();
		_buf = writeBuf;
		_size = buffLen;
	}

	// assign new values
	_str_len = strLen;
}

template<class T>
void axis::AxisString<T>::copy(const char_type * const s, size_t length)
{
	T *writeBuf = _buf;
	size_t buffLen = _size;

	// check if we need to re-allocate space
	if (needs_reallocate_buffer(length, buffLen))
	{
		buffLen = calculate_buffer_size(length);
#if defined AXIS_NO_MEMORY_ARENA
    writeBuf = new T[buffLen];
#else
    writeBuf = (char_type *)System::StringMemory().Allocate(buffLen*sizeof(char_type));
#endif
	}

	if (writeBuf != s)
	{
		try
		{
			for (size_t i = 0; i < length; i++)
			{
				writeBuf[i] = s[i];
			}
		}
		catch (...)
		{	// if something wrong happens, we must delete buffer from the free store
			if (writeBuf != _buf) 
      {
#if defined AXIS_NO_MEMORY_ARENA
        delete [] writeBuf;
#else
        System::StringMemory().Deallocate(writeBuf);
#endif
      }
			throw;
		}
	}
	writeBuf[length] = NULL;

	// if we re-allocated earlier, free old buffer
	if (writeBuf != _buf)
	{
		free_buffer();
		_buf = writeBuf;
		_size = buffLen;
	}

	// assign new values
	_str_len = length;
}

template<class T>
void axis::AxisString<T>::copy(const char_type * const s, size_t length, size_t reserve)
{
	T *writeBuf = _buf;
	size_t buffLen = _size;
	size_t totalSize = length + reserve;

	// check if we need to re-allocate space
	if (needs_reallocate_buffer(totalSize, buffLen))
	{
		buffLen = calculate_buffer_size(totalSize);
#if defined AXIS_NO_MEMORY_ARENA
    writeBuf = new T[buffLen];
#else
    writeBuf = (char_type *)System::StringMemory().Allocate(buffLen*sizeof(char_type));
#endif
	}

	if (writeBuf != s)
	{
		try
		{
			for (size_t i = 0; i < length; i++)
			{
				writeBuf[i] = s[i];
			}
		}
		catch (...)
		{	// if something wrong happens, we must delete buffer from the free store
      if (writeBuf != _buf) 
      {
#if defined AXIS_NO_MEMORY_ARENA
        delete [] writeBuf;
#else
        System::StringMemory().Deallocate(writeBuf);
#endif
      }
			throw;
		}
	}
	writeBuf[length] = NULL;

	// if we re-allocated earlier, free old buffer
	if (writeBuf != _buf)
	{
		free_buffer();
		_buf = writeBuf;
		_size = buffLen;
	}

	// assign new values
	_str_len = length;
}

template<class T>
void axis::AxisString<T>::copy(const char_type * const s)
{
	// check string length
	size_t strLen = 0;
	while(s[strLen] != NULL) strLen++;

	T *writeBuf = _buf;
	size_t buffLen = _size;

	// check if we need to re-allocate space
	if (needs_reallocate_buffer(strLen, buffLen))
	{
		buffLen = calculate_buffer_size(strLen);
#if defined AXIS_NO_MEMORY_ARENA
    writeBuf = new T[buffLen];
#else
    writeBuf = (char_type *)System::StringMemory().Allocate(buffLen*sizeof(char_type));
#endif
	}

	if (writeBuf != s)
	{
		try
		{
			for (size_t i = 0; i < strLen; i++)
			{
				writeBuf[i] = s[i];
			}
		}
		catch (...)
		{	// if something wrong happens, we must delete buffer from the free store
      if (writeBuf != _buf) 
      {
#if defined AXIS_NO_MEMORY_ARENA
        delete [] writeBuf;
#else
        System::StringMemory().Deallocate(writeBuf);
#endif
      }
			throw;
		}
	}
	writeBuf[strLen] = NULL;

	// if we re-allocated earlier, free old buffer
	if (writeBuf != _buf)
	{
		free_buffer();
		_buf = writeBuf;
		_size = buffLen;
	}

	// assign new values
	_str_len = strLen;
}

template<class T>
axis::AxisString<T>::~AxisString(void)
{
	free_buffer();
}

template<class T>
typename axis::AxisString<T>::iterator axis::AxisString<T>::begin( void ) const
{
	return iterator(_buf, _buf + (_str_len == 0? 0 : _str_len - 1));
}

template<class T>
typename axis::AxisString<T>::iterator axis::AxisString<T>::end( void ) const
{
	T *endAddr = _buf + (_str_len == 0? 0 : _str_len - 1);
	return iterator(_buf, endAddr, (_str_len == 0)? endAddr : endAddr + 1);
}

template<class T>
typename axis::AxisString<T>::reverse_iterator axis::AxisString<T>::rbegin( void ) const
{
	return reverse_iterator(_buf, _buf + (_str_len == 0? 0 : _str_len - 1));
}

template<class T>
typename axis::AxisString<T>::reverse_iterator axis::AxisString<T>::rend( void ) const
{
	T *endAddr = _buf + (_str_len == 0? 0 : _str_len - 1);
	return reverse_iterator(_buf, endAddr, (_str_len == 0)? _buf : _buf - 1);
}

template<class T>
typename axis::AxisString<T>::iterator axis::AxisString<T>::cbegin( void ) const
{
	return iterator(_buf, _buf + (_str_len == 0? 0 : _str_len - 1));
}

template<class T>
typename axis::AxisString<T>::iterator axis::AxisString<T>::cend( void ) const
{
	T *endAddr = _buf + (_str_len == 0? 0 : _str_len - 1);
	return iterator(_buf, endAddr, (_str_len == 0)? endAddr : endAddr + 1);
}

template<class T>
typename axis::AxisString<T>::reverse_iterator axis::AxisString<T>::crbegin( void ) const
{
	return reverse_iterator(_buf, _buf + (_str_len == 0? 0 : _str_len - 1));
}

template<class T>
typename axis::AxisString<T>::reverse_iterator axis::AxisString<T>::crend( void ) const
{
	T *endAddr = _buf + (_str_len == 0? 0 : _str_len - 1);
	return reverse_iterator(_buf, endAddr, (_str_len == 0)? _buf : _buf - 1);
}

template<class T>
const T * axis::AxisString<T>::data( void ) const
{
	return c_str();
}
template<class T>
const T * axis::AxisString<T>::c_str( void ) const
{
	return _buf;
}

template<class T>
size_t axis::AxisString<T>::size( void ) const
{
	return _str_len;
}

template<class T>
size_t axis::AxisString<T>::length( void ) const
{
	return _str_len;
}

template<class T>
size_t axis::AxisString<T>::max_size( void ) const
{
	return AXISSTRING_MAXSIZE;
}

template<class T>
void axis::AxisString<T>::resize( size_t n, T c )
{
	if (n < _str_len)
	{
		if (!is_buffer_too_big(n, _size))
		{	// just shrink string without changing buffer
			_buf[n] = NULL;
			_str_len = n;
		}
		else
		{	// buffer is too long; re-allocate it
			copy(_buf, n);
		}
	}
	else if(n > _str_len)
	{	// re-allocate, if needed, and fill in
		size_t oldLen = _str_len;
		if (n > max_size())
		{
			throw axis::foundation::ArgumentException();
		}

		// reallocate buffer
		copy(_buf, _str_len, n-_str_len);
		for(size_t i = _str_len; i < n; ++i)
		{
			_buf[i] = c;
		}
		_buf[n] = NULL;
	}
}
template<class T>
void axis::AxisString<T>::resize( size_t n )
{
	resize(n, (T)NULL);
}
template<class T>
size_t axis::AxisString<T>::capacity( void ) const
{
	return _size;
}
template<class T>
void axis::AxisString<T>::reserve( size_t res_arg /*= 0*/ )
{
	// do nothing if request capacity is less than current
	if (res_arg <= _size) return;

	if (res_arg > max_size())
	{
		throw axis::foundation::ArgumentException();
	}

	// reallocate
	copy(_buf, _str_len, res_arg - _str_len);
}
template<class T>
void axis::AxisString<T>::clear( void )
{
	if (is_buffer_too_big(0, _size))
	{
		copy(_buf, 0);
	}
	else
	{
		_buf[0] = NULL;
		_str_len = 0;
	}
}
template<class T>
bool axis::AxisString<T>::empty( void ) const
{
	return (_str_len == 0);
}
template<class T>
const T& axis::AxisString<T>::at( size_t pos ) const
{
	if (pos >= _str_len)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	return _buf[pos];
}
template<class T>
T& axis::AxisString<T>::at( size_t pos )
{
	if (pos >= _str_len)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	return _buf[pos];
}
template<class T>
const typename T& axis::AxisString<T>::operator[]( size_t pos ) const
{
	return data()[pos];
}
template<class T>
typename T& axis::AxisString<T>::operator[]( size_t pos )
{
	return _buf[pos];
}
template<class T>
size_t axis::AxisString<T>::find( const char_type * pattern, size_t patternSize, size_t startPos, size_t searchLen, bool entireMatch, bool findLast, bool outOfSet /*= false*/ ) const
{
	// trivial case: string to match is greater than analysed
	// string
	if (searchLen < patternSize)
	{
		return self::npos;
	}
	size_t lastFound = self::npos;
	size_t nextPos = startPos;
	size_t stopPos = startPos + searchLen - 1;
	if (stopPos > _str_len - 1)
	{
		throw axis::foundation::ArgumentException();
	}
	while (nextPos <= stopPos && !(lastFound != npos && !findLast))
	{
		if (nextPos + patternSize - 1 > stopPos && entireMatch)
		{	// there is no more matches
			break;
		}
		if (entireMatch)
		{
			if (compare(&_buf[nextPos], patternSize, pattern, patternSize) == 0)
			{
				lastFound = nextPos;
			}
		}
		else
		{
			bool ok = is_one_of(_buf[nextPos], pattern, patternSize);
			if (ok ^ outOfSet)
			{
				lastFound = nextPos;
			}
		}
		nextPos++;
	}
	return lastFound;
}
template<class T>
int axis::AxisString<T>::compare( const char_type *what, size_t what_size, const char_type *to_whom, size_t to_whom_size ) const
{
	size_t compare_size = what_size < to_whom_size? what_size : to_whom_size;
	for (size_t i = 0; i < compare_size; ++i)
	{
		if (what[i] - to_whom[i] != 0)
		{
			return what[i] - to_whom[i];
		}
	}
	if (what_size > to_whom_size)
	{
		return 1;
	}
	else if (to_whom_size > what_size)
	{
		return -1;
	}
	return 0;
}
template<class T>
bool axis::AxisString<T>::is_one_of(char_type what, const char_type *pattern, size_t pattern_size) const
{
	for (size_t i = 0; i < pattern_size; ++i)
	{
		if (what == pattern[i]) return true;
	}
	return false;
}
template<class T>
size_t axis::AxisString<T>::find( const self& str, size_t pos /*= 0 */ ) const
{
	return find(str.data(), str.size(), pos, _str_len, true, false);
}
template<class T>
size_t axis::AxisString<T>::find( const T* s, size_t pos, size_t n ) const
{
	return find(s, n, pos, _str_len, true, false);
}
template<class T>
size_t axis::AxisString<T>::find( const T* s, size_t pos /*= 0 */ ) const
{
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return find(s, len, pos, _str_len, true, false);
}
template<class T>
size_t axis::AxisString<T>::find( T c, size_t pos /*= 0 */ ) const
{
	return find(&c, 1, pos, _str_len, true, false);
}
template<class T>
size_t axis::AxisString<T>::rfind( const self& str, size_t pos /*= 0 */ ) const
{
	return find(str.data(), str.size(), pos, _str_len, true, true);
}
template<class T>
size_t axis::AxisString<T>::rfind( const T* s, size_t pos, size_t n ) const
{
	return find(s, n, pos, _str_len, true, true);
}
template<class T>
size_t axis::AxisString<T>::rfind( const T* s, size_t pos /*= 0 */ ) const
{
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return find(s, len, pos, _str_len, true, true);
}
template<class T>
size_t axis::AxisString<T>::rfind( T c, size_t pos /*= 0 */ ) const
{
	return find(&c, 1, pos, _str_len, true, true);
}
template<class T>
size_t axis::AxisString<T>::find_first_of( const self& str, size_t pos /*= 0 */ ) const
{
	return find(str.data(), str.size(), pos, _str_len, false, false);
}
template<class T>
size_t axis::AxisString<T>::find_first_of( const T* s, size_t pos, size_t n ) const
{
	return find(s, n, pos, _str_len, false, false);
}
template<class T>
size_t axis::AxisString<T>::find_first_of( const T* s, size_t pos /*= 0 */ ) const
{
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return find(s, len, pos, _str_len, false, false);
}
template<class T>
size_t axis::AxisString<T>::find_first_of( T c, size_t pos /*= 0 */ ) const
{
	return find(&c, 1, pos, _str_len, false, false);
}
template<class T>
size_t axis::AxisString<T>::find_last_of( const self& str, size_t pos /*= 0 */ ) const
{
	return find(str.data(), str.size(), pos, _str_len, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_last_of( const T* s, size_t pos, size_t n ) const
{
	return find(s, n, pos, _str_len, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_last_of( const T* s, size_t pos /*= 0 */ ) const
{
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return find(s, len, pos, _str_len, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_last_of( T c, size_t pos /*= 0 */ ) const
{
	return find(&c, 1, pos, _str_len, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_first_not_of( const self& str, size_t pos /*= 0 */ ) const
{
	return find(str.data(), str.size(), pos, _str_len, false, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_first_not_of( const T* s, size_t pos, size_t n ) const
{
	return find(s, n, pos, _str_len, false, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_first_not_of( const T* s, size_t pos /*= 0 */ ) const
{
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return find(s, len, pos, _str_len, false, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_first_not_of( T c, size_t pos /*= 0 */ ) const
{
	return find(&c, 1, pos, _str_len, false, false, true);
}
template<class T>
size_t axis::AxisString<T>::find_last_not_of( const self& str, size_t pos /*= 0 */ ) const
{
	return find(str.data(), str.size(), pos, _str_len, false, true, true);
}
template<class T>
size_t axis::AxisString<T>::find_last_not_of( const T* s, size_t pos, size_t n ) const
{
	return find(s, n, pos, _str_len, false, true, true);
}
template<class T>
size_t axis::AxisString<T>::find_last_not_of( const T* s, size_t pos /*= 0 */ ) const
{
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return find(s, len, pos, _str_len, false, true, true);
}
template<class T>
size_t axis::AxisString<T>::find_last_not_of( T c, size_t pos /*= 0 */ ) const
{
	return find(&c, 1, pos, _str_len, false, true, true);
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::substr( size_t pos /*= 0*/, size_t n /*= npos */ ) const
{
	if (pos > _str_len)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	size_t len = ((pos + n > _str_len) || (n == npos)) ? _str_len - pos : n;
	if (len == 0)
	{	// position after the last char; return an empty string
		return self();
	}
	return self(&_buf[pos], len);
}
template<class T>
int axis::AxisString<T>::compare( const self& str ) const
{
	return compare(_buf, _str_len, str.data(), str.size());
}
template<class T>
int axis::AxisString<T>::compare( const T* s ) const
{
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return compare(_buf, _str_len, s, len);
}
template<class T>
int axis::AxisString<T>::compare( size_t pos1, size_t n1, const self& str ) const
{
	if (pos1 >= _str_len) throw axis::foundation::OutOfBoundsException();
	if (pos1 + n1 > _str_len) n1 = _str_len - pos1;
	return compare(&_buf[pos1], n1, str.data(), str.size());
}
template<class T>
int axis::AxisString<T>::compare( size_t pos1, size_t n1, const T* s ) const
{
	if (pos1 >= _str_len) throw axis::foundation::OutOfBoundsException();
	if (pos1 + n1 > _str_len) n1 = _str_len - pos1;
	// find pattern length
	size_t len = 0;
	while (s[len] != NULL) len++;
	return compare(&_buf[pos1], n1, s, len);
}
template<class T>
int axis::AxisString<T>::compare( size_t pos1, size_t n1, const self& str, size_t pos2, size_t n2 ) const
{
	if (pos1 >= _str_len) throw axis::foundation::OutOfBoundsException();
	if (pos1 + n1 > _str_len) n1 = _str_len - pos1;
	if (pos2 >= str.size()) throw axis::foundation::OutOfBoundsException();
	if (pos2 + n2 > str.size()) n2 = str.size() - pos2;
	return compare(&_buf[pos1], n1, &str.data()[pos2], n2);
}
template<class T>
int axis::AxisString<T>::compare( size_t pos1, size_t n1, const T* s, size_t n2 ) const
{
	if (pos1 >= _str_len) throw axis::foundation::OutOfBoundsException();
	if (pos1 + n1 > _str_len) n1 = _str_len - pos1;
	return compare(&_buf[pos1], n1, s, n2);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( const self& str )
{
	// check if we can insert this string in the blank space
	size_t sz = str.size();
	if (is_buffer_too_small(sz + _str_len, _size))
	{	// reallocate buffer
		copy(_buf, _str_len, sz);
	}

	// append string
	for (size_t i = 0; i < sz; ++i)
	{
		_buf[_str_len + i] = str[i];
	}
	_buf[_str_len + sz] = NULL;
	_str_len += sz;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( const self& str, size_t pos, size_t n )
{
	if (pos + 1 > str.size()) throw axis::foundation::OutOfBoundsException();
	n = pos + n > str.size() ? str.size() - pos : n;

	// check if we can insert this string in the blank space
	if (is_buffer_too_small(n + _str_len, _size))
	{	// reallocate buffer
		copy(_buf, _str_len, n);
	}

	// append string
	for (size_t i = 0; i < n; ++i)
	{
		_buf[_str_len + i] = str[pos+i];
	}
	_buf[_str_len + n] = NULL;
	_str_len += n;
	return *this;
}

template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( const T* s, size_t n )
{
	// check if we can insert this string in the blank space
	if (is_buffer_too_small(_str_len + n, _size))
	{	// reallocate buffer
		copy(_buf, _str_len, n);
	}

	// append string
	for (size_t i = 0; i < n; ++i)
	{
		_buf[_str_len + i] = s[i];
	}
	_buf[_str_len + n] = NULL;
	_str_len += n;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( const T* s )
{
	size_t len = 0;
	while (s[len] != NULL) len++;

	// check if we can insert this string in the blank space
	if (is_buffer_too_small(_str_len + len, _size))
	{	// reallocate buffer
		copy(_buf, _str_len, len);
	}

	// append string
	for (size_t i = 0; i < len; ++i)
	{
		_buf[_str_len + i] = s[i];
	}
	_buf[_str_len + len] = NULL;
	_str_len += len;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( size_t n, T c )
{
	// check if we can insert this string in the blank space
	if (is_buffer_too_small(_str_len + n, _size))
	{	// reallocate buffer
		copy(_buf, _str_len, n);
	}

	// append string
	for (size_t i = 0; i < n; ++i)
	{
		_buf[_str_len + i] = c;
	}
	_buf[_str_len + n] = NULL;
	_str_len += n;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( T c )
{
	return *this += c;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( const_iterator begin, const_iterator end )
{
	size_t n = end.substring_length(begin);
	if (n == 0) return *this;

	// check if we can insert this string in the blank space
	if (is_buffer_too_small(_str_len + n, _size))
	{	// reallocate buffer
		copy(_buf, _str_len, n);
	}
	size_t i = 0;
	while (begin != end)
	{
		_buf[_str_len + i] = *begin;
		++begin;
		++i;
	}
	_buf[_str_len + n] = NULL;
	_str_len += n;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::append( const_reverse_iterator begin, const_reverse_iterator end )
{
	size_t n = end.substring_length(begin);
	if (n == 0) return *this;

	// check if we can insert this string in the blank space
	if (is_buffer_too_small(_str_len + n, _size))
	{	// reallocate buffer
		copy(_buf, _str_len, n);
	}
	size_t i = 0;
	while (begin != end)
	{
		_buf[_str_len + i] = *begin;
		++begin;
		++i;
	}
	_buf[_str_len + n] = NULL;
	_str_len += n;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::operator+=( const self& str )
{
	return append(str);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::operator+=( const T* s )
{
	return append(s);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::operator+=( T c )
{
	return append(1, c);
}
template<class T>
void axis::AxisString<T>::push_back( T c )
{
	append(1, c);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::assign( const self& str )
{
	return (*this = str);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::assign( const self& str, size_t pos, size_t n )
{
	if (pos >= str.size()) throw axis::foundation::OutOfBoundsException();
	if (n > str.size() - pos) n = str.size() - pos;

	copy(&str.c_str()[pos], n);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::assign( const T* s, size_t n )
{
	copy(s, n);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::assign( const T* s )
{
	copy(s);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::assign( size_t n, T c )
{
	copy(&c, 1, n-1);
	for (size_t i = 1; i < n; ++i)
	{
		_buf[i] = c;
	}
	_buf[n] = NULL;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::assign( const_iterator begin, const_iterator end )
{
	size_t n = end.substring_length(begin);
	copy(_buf, 0, n);

	size_t i = 0;
	if (n > 0)
	{
		while (begin != end)
		{
			_buf[i] = *begin;
			++begin;
			++i;
		}
	}
	_buf[i] = NULL;
	_str_len = n;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::assign( const_reverse_iterator begin, const_reverse_iterator end )
{
	size_t n = end.substring_length(begin);
	copy(_buf, 0, n);

	size_t i = 0;
	if (n > 0)
	{
		while (begin != end)
		{
			_buf[i] = *begin;
			++begin;
			++i;
		}
	}
	_buf[i] = NULL;
	_str_len = n;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::insert( size_t pos1, const self& str )
{
	size_t n = str.size();
	return insert(pos1, str, 0, n);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::insert( size_t pos1, const self& str, size_t pos2, size_t n )
{
	if (pos2 >= str.size()) throw axis::foundation::OutOfBoundsException();
	if (pos2 + n > str.size()) n = str.size() - pos2;
	if (pos1 > _str_len) throw axis::foundation::OutOfBoundsException();

	// re-allocate buffer if needed
	copy(_buf, _str_len, n);

	// move tail
	if (_str_len > 0)
	{
		for (size_t i = _str_len - 1; i >= pos1 && i != npos; i--)
		{
			_buf[i + n] = _buf[i];
		}
	}

	// insert new chars
	for (size_t i = 0; i < n; i++)
	{
		_buf[pos1 + i] = str[pos2 + i];
	}
	_str_len += n;
	_buf[_str_len] = NULL;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::insert( size_t pos1, const T* s, size_t n )
{
	if (pos1 > _str_len) throw axis::foundation::OutOfBoundsException();

	// re-allocate buffer if needed
	copy(_buf, _str_len, n);

	// move tail
	if (_str_len > 0)
	{
		for (size_t i = _str_len - 1; i >= pos1 && i != npos; i--)
		{
			_buf[i + n] = _buf[i];
		}
	}

	// insert new chars
	for (size_t i = 0; i < n; i++)
	{
		_buf[pos1 + i] = s[i];
	}
	_str_len += n;
	_buf[_str_len] = NULL;
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::insert( size_t pos1, const T* s )
{
	size_t n = 0;
	while (s[n] != NULL) n++;

	return insert(pos1, s, n);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::insert( size_t pos1, size_t n, T c )
{
	if (pos1 > _str_len) throw axis::foundation::OutOfBoundsException();
	copy(_buf, _str_len, n);

	// move tail
	if (_str_len > 0)
	{
		for (size_t i = _str_len - 1; i >= pos1 && i != npos; i--)
		{
			_buf[i + n] = _buf[i];
		}
	}

	// insert new chars
	for (size_t i = 0; i < n; i++)
	{
		_buf[pos1 + i] = c;
	}
	_str_len += n;
	_buf[_str_len] = NULL;
	return *this;
}
template<class T>
typename axis::AxisString<T>::iterator axis::AxisString<T>::insert( iterator p, T c )
{
	insert(p, 1, c);
	return iterator(_buf, _buf + _str_len - 1, p._curAddr);
}
template<class T>
void axis::AxisString<T>::insert( iterator p, size_t n, T c )
{
	// translate iterator into a relative address
	size_t i = p.substring_length(begin());
	insert(i, n, c);
}
template<class T>
void axis::AxisString<T>::insert( iterator p, const_iterator first, const_iterator last )
{
	size_t n = last.substring_length(first);
	size_t pos = p.substring_length(begin());

	if (n == 0) return;

	// re-allocate buffer if needed
	copy(_buf, _str_len, n);

	// move tail
	if (_str_len > 0)
	{
		for (size_t i = _str_len - 1; i >= pos && i != npos; i--)
		{
			_buf[i + n] = _buf[i];
		}
	}

	// insert new chars
	size_t i = 0;
	while(first != last)
	{
		_buf[pos + i] = *first;
		++first;
		++i;
	}
	_str_len += n;
	_buf[_str_len] = NULL;
}
template<class T>
void axis::AxisString<T>::insert( iterator p, const_reverse_iterator first, const_reverse_iterator last )
{
	size_t n = last.substring_length(first);
	size_t pos = p.substring_length(begin());

	if (n == 0) return;

	// re-allocate buffer if needed
	copy(_buf, _str_len, n);

	// move tail
	if (_str_len > 0)
	{
		for (size_t i = _str_len - 1; i >= pos && i != npos; i--)
		{
			_buf[i + n] = _buf[i];
		}
	}

	// insert new chars
	size_t i = 0;
	while(first != last)
	{
		_buf[pos + i] = *first;
		++first;
		++i;
	}
	_str_len += n;
	_buf[_str_len] = NULL;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::erase( size_t pos /*= 0*/, size_t n /*= npos */ )
{
	if (n > _str_len - pos) n = _str_len - pos;

	// erase doesn't free space, since operation's use is rare
	// for freeing space, clear() is more often used
	size_t tailLen = _str_len - pos - n;
	for (size_t i = 0; i < tailLen; i++)
	{
		_buf[pos+i] = _buf[pos+i+n];
	}
	_str_len -= n;
	_buf[_str_len] = NULL;
	return *this;
}
template<class T>
typename axis::AxisString<T>::iterator axis::AxisString<T>::erase( iterator position )
{
	size_t pos = position.substring_length(begin());
	erase(pos);
	return iterator(_buf, _buf + _str_len, _buf + pos);
}
template<class T>
typename axis::AxisString<T>::iterator axis::AxisString<T>::erase( iterator first, iterator last )
{
	size_t pos = first.substring_length(begin());
	size_t n = last.substring_length(first);
	erase(pos, n);
	return iterator(_buf, _buf + _str_len, _buf + pos);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( size_t pos1, size_t n1, const self& str )
{
	return replace(pos1, n1, str, 0, str.size());
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( iterator i1, iterator i2, const self& str )
{
	size_t pos = i1.substring_length(begin());
	size_t n = i2.substring_length(i1);
	return replace(pos, n, str);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( size_t pos1, size_t n1, const self& str, size_t pos2, size_t n2 )
{
  if (n2 != 0)
  {
    if (pos2 >= str.size()) throw axis::foundation::OutOfBoundsException();
    if (n2 + pos2 > str.size()) n2 = str.size() - pos2;
    return replace(pos1, n1, &str.data()[pos2], n2);
  }
  else
  {
    if (pos2 != 0) throw axis::foundation::OutOfBoundsException();
    if (n2 > str.size()) n2 = str.size();
    char_type c = NULL;
    return replace(pos1, n1, &c, 0);
  }
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( size_t pos1, size_t n1, const T* s, size_t n2 )
{
	if (pos1 >= _str_len) throw axis::foundation::OutOfBoundsException();
	if (pos1 + n1 > _str_len) n1 = _str_len - n1;

	// re-allocate buffer if needed
	if (n2 > n1)
	{
    copy(_buf, _str_len, n2-n1);
	}

	size_t tailLen = _str_len - n1 - pos1;
	if (n1 > n2)	// pull tail
	{
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[pos1 + n2 + i] = _buf[pos1 + n1 + i];
		}
	}
	else if (n2 > n1)
	{
		size_t offset = n2-n1;
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[_str_len+offset-i-1] = _buf[_str_len-i-1];
		}
	}
	// insert new string into the specified position
	for (size_t i = 0; i < n2; i++)
	{
		_buf[pos1+i] = s[i];
	}
	_buf[_str_len+n2-n1] = NULL;
	_str_len += (n2-n1);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( iterator i1, iterator i2, const T* s, size_t n2 )
{
	size_t pos = i1.substring_length(begin());
	size_t n = i2.substring_length(i1);
	return replace(pos, n, s, n2);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( size_t pos1, size_t n1, const T* s )
{
	// find string length
	size_t n2 = 0;
	while (s[n2] != NULL) n2++;
	return replace(pos1, n1, s, n2);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( iterator i1, iterator i2, const T* s )
{
	size_t pos = i1.substring_length(begin());
	size_t n = i2.substring_length(i1);

	// find string length
	size_t n2 = 0;
	while (s[n2] != NULL) n2++;

	return replace(pos, n, s, n2);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( size_t pos1, size_t n1, size_t n2, T c )
{
	if (pos1 >= _str_len) throw axis::foundation::OutOfBoundsException();
	if (pos1 + n1 > _str_len) n1 = _str_len - n1;

	// re-allocate buffer if needed
	copy(_buf, _str_len, n2-n1);

	size_t tailLen = _str_len - n1 - pos1;
	if (n1 > n2)	// pull tail
	{
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[pos1 + n2 + i] = _buf[pos1 + n1 + i];
		}
	}
	else if (n2 > n1)
	{
		size_t offset = n2-n1;
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[_str_len+offset-i-1] = _buf[_str_len-i-1];
		}
	}
	// insert new string into the specified position
	for (size_t i = 0; i < n2; i++)
	{
		_buf[pos1+i] = c;
	}
	_buf[_str_len+n2-n1] = NULL;
	_str_len += (n2-n1);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( iterator i1, iterator i2, size_t n2, T c )
{
	size_t pos = i1.substring_length(begin());
	size_t n = i2.substring_length(i1);

	return replace(pos, n, n2, c);
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( iterator from, iterator to, iterator src_from, iterator src_end )
{
	size_t pos1 = from.substring_length(begin());
	size_t n1 = to.substring_length(from);
	size_t n2 = src_end.substring_length(src_from);

	// re-allocate buffer if needed
	copy(_buf, _str_len, n2-n1);

	size_t tailLen = _str_len - n1 - pos1;
	if (n1 > n2)	// pull tail
	{
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[pos1 + n2 + i] = _buf[pos1 + n1 + i];
		}
	}
	else if (n2 > n1)
	{
		size_t offset = n2-n1;
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[_str_len+offset-i-1] = _buf[_str_len-i-1];
		}
	}
	// insert new string into the specified position
	for (size_t i = 0; i < n2; i++)
	{
		_buf[pos1+i] = *src_from;
		++src_from;
	}
	_buf[_str_len+n2-n1] = NULL;
	_str_len += (n2-n1);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( iterator from, iterator to, reverse_iterator src_from, reverse_iterator src_end )
{
	size_t pos1 = from.substring_length(begin());
	size_t n1 = to.substring_length(from);
	size_t n2 = src_end.substring_length(src_from);

	// re-allocate buffer if needed
	copy(_buf, _str_len, n2-n1);

	size_t tailLen = _str_len - n1 - pos1;
	if (n1 > n2)	// pull tail
	{
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[pos1 + n2 + i] = _buf[pos1 + n1 + i];
		}
	}
	else if (n2 > n1)
	{
		size_t offset = n2-n1;
		for (size_t i = 0; i < tailLen; i++)
		{
			_buf[_str_len+offset-i-1] = _buf[_str_len-i-1];
		}
	}
	// insert new string into the specified position
	for (size_t i = 0; i < n2; i++)
	{
		_buf[pos1+i] = *src_from;
		++src_from;
	}
	_buf[_str_len+n2-n1] = NULL;
	_str_len += (n2-n1);
	return *this;
}

template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::replace( const self& what, const self& s )
{
	size_t pos = find(what);
	if (pos == npos) return *this;
	return replace(pos, what.size(), s);
}

template<class T>
void axis::AxisString<T>::swap( self& str )
{
	// change buffer information
	_buf = (char_type *)((size_t)_buf ^ (size_t)str._buf);				// XOR swap algorithm
	str._buf = (char_type *)((size_t)_buf ^ (size_t)str._buf);
	_buf = (char_type *)((size_t)_buf ^ (size_t)str._buf);

	_size ^= str._size;
	str._size = _size ^ str._size;
	_size ^= str._size;

	_str_len ^=str._str_len;
	str._str_len = _str_len ^ str._str_len;
	_str_len ^= str._str_len;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::reverse( void )
{
	size_t len = _str_len / 2;
	for (size_t i = 0; i < len; i++)
	{
		_buf[i] ^= _buf[_str_len - i - 1];
		_buf[_str_len - i - 1] ^= _buf[i];
		_buf[i] ^= _buf[_str_len - i - 1];
	}
	return *this;
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::operator+( const self& s ) const
{
	return self(*this).append(s);
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::operator + (const T *s) const
{
	return self(*this).append(s);
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::operator + (T c) const
{
	return self(*this).append(c);
}

template<>
axis::AxisString<char>::self& axis::AxisString<char>::trim_left( void )
{
	// find first non-blank character
	size_t pos = find_first_not_of(" \t\n\r\0");
	if (pos == npos) return *this;
	erase(0, pos);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::trim_left( void )
{
	// find first non-blank character
	size_t pos = find_first_not_of(_T(" \t\n\r\0"));
	if (pos == npos) return *this;
	erase(0, pos);
	return *this;
}
template<>
typename axis::AxisString<char>::self& axis::AxisString<char>::trim_right( void )
{
	// find first non-blank character
	size_t pos = find_first_not_of(" \t\n\r\0");
	if (pos == npos) pos = 0;
	pos = find_first_of(" \t\n\r\0", pos);
	if (pos == npos) return *this;
	erase(pos);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::trim_right( void )
{
	// find last non-blank character
	size_t pos = find_last_not_of(_T(" \t\n\r\0"));
	if (pos == npos || pos == _str_len - 1) return *this;
	erase(pos + 1);
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::trim( void )
{
	trim_left();
	return trim_right();
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::to_lower_case( void )
{
	for (size_t i = 0; i < _str_len; i++)
	{
		if ((_buf[i] >= 'A' && _buf[i] <= 'Z'))
		{
			_buf[i] -= 'A';
			_buf[i] += 'a';
		}
	}
	return *this;
}
template<class T>
typename axis::AxisString<T>::self& axis::AxisString<T>::to_upper_case( void )
{
	for (size_t i = 0; i < _str_len; i++)
	{
		if ((_buf[i] >= 'a' && _buf[i] <= 'z'))
		{
			_buf[i] -= 'a';
			_buf[i] += 'A';
		}
	}
	return *this;
}
template<class T>
bool axis::AxisString<T>::equals( const self& s ) const
{
	return compare(s) == 0;
}
template<class T>
bool axis::AxisString<T>::equals_case_insensitive( const self& s ) const
{
	size_t compare_size = _str_len < s.size()? _str_len : s.size();
	for (size_t i = 0; i < compare_size; ++i)
	{
		if (_buf[i] - s[i] != 0)
		{
			// recheck for insensitive case
			if (!(((_buf[i] >= 'a' && _buf[i] <= 'z') || (_buf[i] >= 'A' && _buf[i] <= 'Z')) &&
				((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z'))))
			{
				return (_buf[i] - s[i]) == 0;
			}
			T myChar = (_buf[i] >= 'a' && _buf[i] <= 'z')? _buf[i]-'a' : _buf[i]-'A';
			T otherChar = (s[i] >= 'a' && s[i] <= 'z')? s[i]-'a' : s[i]-'A';
			if (myChar != otherChar)
			{
				return (myChar - otherChar) == 0;
			}
		}
	}
	if (_str_len != s.size())
	{
		return false;
	}
	return true;
}
template<class T>
bool axis::AxisString<T>::equals( const T *s ) const
{
	return equals(self(s));
}
template<class T>
bool axis::AxisString<T>::equals_case_insensitive( const T *s ) const
{
	return equals_case_insensitive(self(s));
}
template<class T>
bool axis::AxisString<T>::operator<( const self& s ) const
{
	return compare(s) < 0;
}
template<class T>
bool axis::AxisString<T>::operator>( const self& s ) const
{
	return compare(s) > 0;
}
template<class T>
bool axis::AxisString<T>::operator<=( const self& s ) const
{
	return compare(s) <= 0;
}
template<class T>
bool axis::AxisString<T>::operator>=( const self& s ) const
{
	return compare(s) >= 0;
}
template<class T>
bool axis::AxisString<T>::operator==( const self& s ) const
{
	return equals(s);
}
template<class T>
bool axis::AxisString<T>::operator!=( const self& s ) const
{
	return !equals(s);
}
template<class T>
bool axis::AxisString<T>::operator!=( const T *s ) const
{
	return !equals(s);
}
template<class T>
bool axis::AxisString<T>::operator==( const T *s ) const
{
	return equals(s);
}
template<class T>
axis::AxisString<T>::operator const T*( void ) const
{
	return data();
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::bigint_parse( unsigned long number )
{
	return self(boost::lexical_cast<std::basic_string<T>>(number).c_str());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::bigint_parse(unsigned long number, int numDigits)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.width(numDigits);
	
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::bigint_to_hex(unsigned long number, int numDigits)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.width(numDigits);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::hex, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::bigint_to_octal(unsigned long number, int numDigits)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.width(numDigits);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::oct, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::bigint_to_hex(unsigned long number)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::hex, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::bigint_to_octal(unsigned long number)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::oct, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::int_parse( long number )
{
	return self(boost::lexical_cast<std::basic_string<T>>(number).c_str());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::int_parse(long number, int numDigits)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.width(numDigits);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::int_to_hex(long number, int numDigits)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.width(numDigits);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::hex, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::int_to_octal(long number, int numDigits)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.width(numDigits);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::oct, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::int_to_hex(long number)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::hex, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::int_to_octal(long number)
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.setf(std::ios::oct, std::ios::basefield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::double_parse( float number )
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::floatfield);
	buf.precision(fp_parse_precision);
	buf.setf(std::ios::scientific, std::ios::floatfield);
	buf <<	number;
	return self(buf.str().data());
}
template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::double_parse( double number )
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::floatfield);
	buf.precision(fp_parse_precision);
	buf.setf(std::ios::scientific, std::ios::floatfield);
	buf <<	number;
	return self(buf.str().data());
}

template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::align_left( size_t fieldSize ) const
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.setf(std::ios::left, std::ios::adjustfield);
	buf.width(fieldSize);
	buf << data();
	return self(buf.str().data());
}

template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::align_right( size_t fieldSize ) const
{
	std::basic_ostringstream<T> buf;
	buf.unsetf(std::ios::adjustfield);
	buf.setf(std::ios::right, std::ios::adjustfield);
	buf.width(fieldSize);
	buf << data();
	return self(buf.str().data());
}

template<class T>
typename axis::AxisString<T>::self axis::AxisString<T>::align_center( size_t fieldSize ) const
{
	size_t trailingLeftSize = (fieldSize - size()) / 2;
	size_t trailingRightSize = fieldSize - trailingLeftSize - size();
	if (trailingLeftSize < 0) return *this;

	self buf(trailingLeftSize, ' ');
	buf += *this + self(trailingRightSize, ' ');
	return buf;
}





template<class T>
axis::AxisString<T>::string_iterator::string_iterator( pointer baseAddress, pointer endAddress )
{
	_baseAddr = baseAddress;
	_endAddr = endAddress;
	_curAddr = baseAddress;
}
template<class T>
axis::AxisString<T>::string_iterator::string_iterator( pointer baseAddress, pointer endAddress, pointer currentPosition )
{
	_baseAddr = baseAddress;
	_endAddr = endAddress;
	_curAddr = currentPosition;
}
template<class T>
axis::AxisString<T>::string_iterator::string_iterator( const self& it )
{
	_baseAddr = it._baseAddr;
	_endAddr = it._endAddr;
	_curAddr = it._curAddr;
}
template<class T>
typename axis::AxisString<T>::string_iterator::const_reference axis::AxisString<T>::string_iterator::operator*( void ) const
{
	return *_curAddr;
}
template<class T>
typename axis::AxisString<T>::string_iterator::reference axis::AxisString<T>::string_iterator::operator*( void )
{
	return *_curAddr;
}
template<class T>
typename axis::AxisString<T>::string_iterator::pointer axis::AxisString<T>::string_iterator::operator->( void )
{
	return _curAddr;
}
template<class T>
const typename axis::AxisString<T>::string_iterator::pointer axis::AxisString<T>::string_iterator::operator->( void ) const
{
	return _curAddr;
}
template<class T>
typename axis::AxisString<T>::string_iterator::self& axis::AxisString<T>::string_iterator::operator=( const self& it )
{
	_baseAddr = it._baseAddr;
	_endAddr = it._endAddr;
	_curAddr = it._curAddr;
	return *this;
}

template<class T>
typename axis::AxisString<T>::string_iterator::self& axis::AxisString<T>::string_iterator::operator++( void )
{
	if (_curAddr <= _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::string_iterator::self& axis::AxisString<T>::string_iterator::operator++( void ) const
{
	if (_curAddr <= _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::string_iterator::self axis::AxisString<T>::string_iterator::operator++( int ) const
{
	self it = *this;
	if (_curAddr <= _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::string_iterator::self axis::AxisString<T>::string_iterator::operator++( int )
{
	self it = *this;
	if (_curAddr <= _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::string_iterator::self& axis::AxisString<T>::string_iterator::operator--( void )
{
	if (_curAddr > _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::string_iterator::self& axis::AxisString<T>::string_iterator::operator--( void ) const
{
	if (_curAddr > _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::string_iterator::self axis::AxisString<T>::string_iterator::operator--( int ) const
{
	self it = *this;
	if (_curAddr > _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::string_iterator::self axis::AxisString<T>::string_iterator::operator--( int )
{
	self it = *this;
	if (_curAddr > _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::string_iterator::container axis::AxisString<T>::string_iterator::to_string( const self& it ) const
{
	return container(*this, it);
}
template<class T>
typename axis::AxisString<T>::string_iterator::difference_type axis::AxisString<T>::string_iterator::operator -( const self& it ) const
{
	if (_baseAddr != it._baseAddr) throw axis::foundation::ArgumentException();
	return (difference_type)(((size_t)_curAddr - (size_t)it._curAddr) / sizeof(T));
}
template<class T>
bool axis::AxisString<T>::string_iterator::operator==( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_curAddr == it._curAddr) &&
		   (_endAddr == it._endAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::string_iterator::operator!=( const self& it ) const
{
	return !(*this == it);
}
template<class T>
bool axis::AxisString<T>::string_iterator::operator<=( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		   (_curAddr <= it._curAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::string_iterator::operator>=( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		(_curAddr >= it._curAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::string_iterator::operator<( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		(_curAddr < it._curAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::string_iterator::operator>( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		(_curAddr > it._curAddr) && (_baseAddr != NULL);
}

template<class T>
axis::AxisString<T>::reverse_string_iterator::reverse_string_iterator( pointer baseAddress, pointer endAddress )
{
	_baseAddr = baseAddress;
	_endAddr = endAddress;
	_curAddr = endAddress;
}
template<class T>
axis::AxisString<T>::reverse_string_iterator::reverse_string_iterator( pointer baseAddress, pointer endAddress, pointer currentPosition )
{
	_baseAddr = baseAddress;
	_endAddr = endAddress;
	_curAddr = currentPosition;
}
template<class T>
axis::AxisString<T>::reverse_string_iterator::reverse_string_iterator( const self& it )
{
	_baseAddr = it._baseAddr;
	_endAddr = it._endAddr;
	_curAddr = it._curAddr;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::const_reference axis::AxisString<T>::reverse_string_iterator::operator*( void ) const
{
	return *_curAddr;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::reference axis::AxisString<T>::reverse_string_iterator::operator*( void )
{
	return *_curAddr;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::pointer axis::AxisString<T>::reverse_string_iterator::operator->( void )
{
	return _curAddr;
}
template<class T>
const typename axis::AxisString<T>::reverse_string_iterator::pointer axis::AxisString<T>::reverse_string_iterator::operator->( void ) const
{
	return _curAddr;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self& axis::AxisString<T>::reverse_string_iterator::operator=( const self& it )
{
	_baseAddr = it._baseAddr;
	_endAddr = it._endAddr;
	_curAddr = it._curAddr;
	return *this;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self& axis::AxisString<T>::reverse_string_iterator::operator++( void )
{
	if (_curAddr >= _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::reverse_string_iterator::self& axis::AxisString<T>::reverse_string_iterator::operator++( void ) const
{
	if (_curAddr >= _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::reverse_string_iterator::self axis::AxisString<T>::reverse_string_iterator::operator++( int ) const
{
	self it = *this;
	if (_curAddr >= _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self axis::AxisString<T>::reverse_string_iterator::operator++( int )
{
	self it = *this;
	if (_curAddr >= _baseAddr)
		--_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self& axis::AxisString<T>::reverse_string_iterator::operator--( void )
{
	if (_curAddr < _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::reverse_string_iterator::self& axis::AxisString<T>::reverse_string_iterator::operator--( void ) const
{
	if (_curAddr < _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return *this;
}
template<class T>
const typename axis::AxisString<T>::reverse_string_iterator::self axis::AxisString<T>::reverse_string_iterator::operator--( int ) const
{
	self it = *this;
	if (_curAddr < _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self axis::AxisString<T>::reverse_string_iterator::operator--( int )
{
	self it = *this;
	if (_curAddr < _endAddr)
		++_curAddr;
	else
		throw axis::foundation::InvalidOperationException();
	return it;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::container axis::AxisString<T>::reverse_string_iterator::to_string( const self& it ) const
{
	return container(*this, it);
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::difference_type axis::AxisString<T>::reverse_string_iterator::operator -( const self& it ) const
{
	if (_baseAddr != it._baseAddr) throw axis::foundation::ArgumentException();
	return (difference_type)((size_t)_curAddr - (size_t)it._curAddr) / sizeof(T);
}
template<class T>
bool axis::AxisString<T>::reverse_string_iterator::operator==( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_curAddr == it._curAddr) &&
		(_endAddr == it._endAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::reverse_string_iterator::operator!=( const self& it ) const
{
	return !(*this == it);
}
template<class T>
bool axis::AxisString<T>::reverse_string_iterator::operator<=( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		(_curAddr >= it._curAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::reverse_string_iterator::operator>=( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		(_curAddr <= it._curAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::reverse_string_iterator::operator<( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		(_curAddr > it._curAddr) && (_baseAddr != NULL);
}
template<class T>
bool axis::AxisString<T>::reverse_string_iterator::operator>( const self& it ) const
{
	return (_baseAddr == it._baseAddr) && (_endAddr == it._endAddr) &&
		(_curAddr < it._curAddr) && (_baseAddr != NULL);
}
template<class T>
size_t axis::AxisString<T>::string_iterator::substring_length( const self& it ) const
{
	if (_baseAddr != it._baseAddr) throw axis::foundation::ArgumentException();
	if (_curAddr < it._curAddr) return 0;
	return ((size_t)_curAddr - (size_t)it._curAddr) / sizeof(T);
}
template<class T>
size_t axis::AxisString<T>::reverse_string_iterator::substring_length( const self& it ) const
{
	if (_baseAddr != it._baseAddr) throw axis::foundation::ArgumentException();
	if (_curAddr > it._curAddr) return 0;
	return ((size_t)it._curAddr - (size_t)_curAddr) / sizeof(T);
}
template<class T>
axis::AxisString<T>::string_iterator::string_iterator( void )
{
	_baseAddr = NULL;
	_curAddr = NULL;
	_endAddr = NULL;
}

template<class T>
axis::AxisString<T>::reverse_string_iterator::reverse_string_iterator( void )
{
	_baseAddr = NULL;
	_curAddr = NULL;
	_endAddr = NULL;
}

template<class T>
typename axis::AxisString<T>::string_iterator::self axis::AxisString<T>::string_iterator::operator + (distance_type n)
{
	string_iterator it(*this);
	it += n;
	return it;
}

template<class T>
typename axis::AxisString<T>::string_iterator::self& axis::AxisString<T>::string_iterator::operator += (distance_type n)
{
	if (_curAddr + n > _endAddr + 1)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	_curAddr += n;
	return *this;
}

template<class T>
typename axis::AxisString<T>::string_iterator::self axis::AxisString<T>::string_iterator::operator - (distance_type n)
{
	string_iterator it(*this);
	it -= n;
	return it;
}

template<class T>
typename axis::AxisString<T>::string_iterator::self& axis::AxisString<T>::string_iterator::operator -= (distance_type n)
{
	if (_curAddr - n < _baseAddr)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	_curAddr -= n;
	return *this;
}

template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self axis::AxisString<T>::reverse_string_iterator::operator + (distance_type n)
{
	reverse_string_iterator it(*this);
	it += n;
	return it;
}
template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self& axis::AxisString<T>::reverse_string_iterator::operator += (distance_type n)
{
	if (_curAddr - n < _baseAddr - 1)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	_curAddr -= n;
	return *this;
}

template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self axis::AxisString<T>::reverse_string_iterator::operator - (distance_type n)
{
	reverse_string_iterator it(*this);
	it -= n;
	return it;
}

template<class T>
typename axis::AxisString<T>::reverse_string_iterator::self& axis::AxisString<T>::reverse_string_iterator::operator -= (distance_type n)
{
	if (_curAddr + n > _endAddr)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	_curAddr += n;
	return *this;
}

axis::UnicodeString operator +(const wchar_t *s1, const axis::UnicodeString& s2)
{
	return axis::UnicodeString(s1) + s2;
}
axis::UnicodeString operator +(wchar_t s1, const axis::UnicodeString& s2)
{
	return axis::UnicodeString(1, s1) + s2;
}

axis::AsciiString operator +(const char *s1, const axis::AsciiString& s2)
{
	return axis::AsciiString(s1) + s2;
}
axis::AsciiString operator +(char s1, const axis::AsciiString& s2)
{
	return axis::AsciiString(1, s1) + s2;
}

char_type * axis::StringEncoding::ASCIIToUnicode(const char * s)
{
  int bufLen = (int)strlen(s) + 1;
  char_type * converted = new char_type[bufLen];
#ifdef _UNICODE
  MultiByteToWideChar(CP_ACP, 0, s, -1, converted, bufLen);
#else
  memcpy_s(*converted, s, bufLen);
#endif
  return converted;
}

char * axis::StringEncoding::UnicodeToASCII(const char_type * s)
{
  int bufLen = (int)_tcslen(s) + 1;
  char * converted = new char[bufLen];
#ifdef _UNICODE
  WideCharToMultiByte(CP_ACP, 0, s, -1, converted, bufLen, NULL, NULL);
#else
  memcpy_s(*converted, s, bufLen);
#endif
  return converted;
}

void axis::StringEncoding::AssignFromASCII( const char * s, String& out )
{
#ifdef _UNICODE
  int bufLen = (int)strlen(s) + 1;
  char_type * converted = new char_type[bufLen];
  MultiByteToWideChar(CP_ACP, 0, s, -1, converted, bufLen);
  out = converted;
  delete [] converted;
#else
  out = converted;
#endif
}

void axis::StringServices::Trim( axis::String& out, axis::String& in )
{
  out.clear();

  // return immediately if in is zero-length
  if (in.size() == 0) return;

  String::size_type i = 0; 
  String::size_type start, end;

  while(in[i] == _T(' ') || in[i] == _T('\t'))
  {
    i++;
    if (i >= in.size())
    {	// it is a blank string
      return;
    }
  }
  start = i;

  i = in.size() - 1;
  while(in[i] == _T(' ') || in[i] == _T('\t'))
  {
    i--;
  }
  end = i;
  out = in.substr(start, end - start + 1);
}

bool axis::StringCompareLessThan::operator()( const String& a, const String& b ) const
{
  return a.compare(b) < 0;
}

bool axis::StringCompareGreaterThan::operator()( const String& a, const String& b ) const
{
  return a.compare(b) > 0;
}

bool axis::StringCompareLessOrEqual::operator()( const String& a, const String& b ) const
{
  return a.compare(b) <= 0;
}

bool axis::StringCompareGreaterOrEqual::operator()( const String& a, const String& b ) const
{
  return a.compare(b) >= 0;
}

bool axis::StringCompareEqual::operator()( const String& a, const String& b ) const
{
  return a.compare(b) == 0;
}



uint64 axis::hash_value( const AsciiString& str )
{
  return boost::hash_range(str.begin(), str.end());
}

uint64 axis::hash_value( const UnicodeString& str )
{
  return boost::hash_range(str.begin(), str.end());
}

// explicit instantiate template classes
template class axis::AxisString<wchar_t>;
template class axis::AxisString<char>;
