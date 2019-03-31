/// <summary>
/// Defines the basic String type used in this API and
/// defines auxiliary classes and functions for string conversion.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once

#include <tchar.h>
#include "foundation/Axis.SystemBase.hpp"

namespace axis {

/// <summary>
/// Describes a character sequence and provides operations to work with it.
/// </summary>
template<class T>
class AXISSYSTEMBASE_API AxisString
{
public:
	class AXISSYSTEMBASE_API string_iterator
	{
	public:
		typedef typename T value_type;
		typedef int difference_type;
		typedef size_t distance_type;	// retained
		typedef typename T* pointer;
		typedef typename T& reference;
		typedef const reference const_reference;
		typedef AxisString<value_type> container;
		typedef string_iterator self;

		string_iterator(const self& it);
		string_iterator(void);

		const_reference operator *(void) const;
		reference operator *(void);
		pointer operator ->(void);
		const pointer operator ->(void) const;

		self& operator=(const self& it);

		self& operator++(void);				// prefix
		const self& operator++(void) const;	// prefix
		const self operator++(int) const;
		self operator++(int);

		self& operator--(void);				// prefix
		const self& operator--(void) const;	// prefix
		const self operator--(int) const;
		self operator--(int);

		container to_string(const self& it) const;
		difference_type operator -(const self& it) const;
		size_t substring_length(const self& it) const;

		self operator + (distance_type n);
		self& operator += (distance_type n);
		self operator - (distance_type n);
		self& operator -= (distance_type n);

		bool operator ==(const self& it) const;
		bool operator !=(const self& it) const;
		bool operator <=(const self& it) const;
		bool operator >=(const self& it) const;
		bool operator <(const self& it) const;
		bool operator >(const self& it) const;

		template<class T>
		friend class AxisString;
	private:
		mutable pointer _curAddr;
		pointer _baseAddr;
		pointer _endAddr;

		string_iterator(pointer baseAddress, pointer endAddress);
		string_iterator(pointer baseAddress, pointer endAddress, pointer currentPosition);
	};

	class AXISSYSTEMBASE_API reverse_string_iterator
	{
	public:
		typedef typename T value_type;
		typedef int difference_type;
		typedef size_t distance_type;	// retained
		typedef typename T* pointer;
		typedef typename T& reference;
		typedef const reference const_reference;
		typedef AxisString<value_type> container;
		typedef reverse_string_iterator self;

		reverse_string_iterator(const self& it);
		reverse_string_iterator(void);

		const_reference operator *(void) const;
		reference operator *(void);
		pointer operator ->(void);
		const pointer operator ->(void) const;

		self& operator=(const self& it);

		self& operator++(void);				// prefix
		const self& operator++(void) const;	// prefix
		const self operator++(int) const;
		self operator++(int);

		self operator + (distance_type n);
		self& operator += (distance_type n);
		self operator - (distance_type n);
		self& operator -= (distance_type n);

		self& operator--(void);				// prefix
		const self& operator--(void) const;	// prefix
		const self operator--(int) const;
		self operator--(int);

		container to_string(const self& it) const;
		difference_type operator -(const self& it) const;
		size_t substring_length(const self& it) const;

		bool operator ==(const self& it) const;
		bool operator !=(const self& it) const;
		bool operator <=(const self& it) const;
		bool operator >=(const self& it) const;
		bool operator <(const self& it) const;
		bool operator >(const self& it) const;

		template<class T>
		friend class AxisString;
	private:
		mutable pointer _curAddr;
		pointer _baseAddr;
		pointer _endAddr;

		reverse_string_iterator(pointer baseAddress, pointer endAddress);
		reverse_string_iterator(pointer baseAddress, pointer endAddress, pointer currentPosition);
	};

	typedef string_iterator iterator;
	typedef reverse_string_iterator reverse_iterator;
	typedef const iterator const_iterator;
	typedef const reverse_iterator const_reverse_iterator;

	typedef typename T char_type;
	typedef typename T value_type;
	typedef size_t size_type;
	typedef AxisString<typename T> self;

  static const size_t npos;
  static const unsigned int fp_parse_precision;

	AxisString(void);
	AxisString(const self& str);
	AxisString(const self& str, size_t pos, size_t n = npos);
	AxisString(const T *s, size_t n);
	AxisString(const T *s);
	AxisString(size_t n, T s);
	explicit AxisString(T s);
	AxisString(const_iterator& begin, const_iterator& end);
	AxisString(const_reverse_iterator& begin, const_reverse_iterator& end);
	self& operator =(const self& s);
	self& operator =(const T *s);
	self& operator =(const T c);

	~AxisString(void);

  static self bigint_parse(unsigned long number);
  static self bigint_parse(unsigned long number, int numDigits);
  static self bigint_to_hex(unsigned long number, int numDigits);
  static self bigint_to_octal(unsigned long number, int numDigits);
  static self bigint_to_hex(unsigned long number);
  static self bigint_to_octal(unsigned long number);
  static self int_parse(long number);
  static self int_parse(long number, int numDigits);
  static self int_to_hex(long number, int numDigits);
  static self int_to_octal(long number, int numDigits);
  static self int_to_hex(long number);
  static self int_to_octal(long number);
  static self double_parse(double number);
  static self double_parse(float number);

	iterator begin(void) const;
	iterator end(void) const;
	reverse_iterator rbegin(void) const;
	reverse_iterator rend(void) const;

	iterator cbegin(void) const;
	iterator cend(void) const;
	reverse_iterator crbegin(void) const;
	reverse_iterator crend(void) const;

	const T *data(void) const;
	const T *c_str(void) const;

	size_t size(void) const;
	size_t length(void) const;
	size_t max_size(void) const;

	void resize(size_t n, T c);
	void resize(size_t n);
	size_t capacity(void) const;
	void reserve(size_t res_arg = 0);

	void clear(void);
	bool empty(void) const;

	const T& at(size_t pos) const;
	T& at(size_t pos);

	const T& operator[] (size_t pos) const;
	T& operator[] (size_t pos);

	size_t find ( const self& str, size_t pos = 0 ) const;
	size_t find ( const T* s, size_t pos, size_t n ) const;
	size_t find ( const T* s, size_t pos = 0 ) const;
	size_t find ( T c, size_t pos = 0 ) const;

	size_t rfind ( const self& str, size_t pos = 0 ) const;
	size_t rfind ( const T* s, size_t pos, size_t n ) const;
	size_t rfind ( const T* s, size_t pos = 0 ) const;
	size_t rfind ( T c, size_t pos = 0 ) const;

	size_t find_first_of ( const self& str, size_t pos = 0 ) const;
	size_t find_first_of ( const T* s, size_t pos, size_t n ) const;
	size_t find_first_of ( const T* s, size_t pos = 0 ) const;
	size_t find_first_of ( T c, size_t pos = 0 ) const;

	size_t find_last_of ( const self& str, size_t pos = 0 ) const;
	size_t find_last_of ( const T* s, size_t pos, size_t n ) const;
	size_t find_last_of ( const T* s, size_t pos = 0 ) const;
	size_t find_last_of ( T c, size_t pos = 0 ) const;

	size_t find_first_not_of ( const self& str, size_t pos = 0 ) const;
	size_t find_first_not_of ( const T* s, size_t pos, size_t n ) const;
	size_t find_first_not_of ( const T* s, size_t pos = 0 ) const;
	size_t find_first_not_of ( T c, size_t pos = 0 ) const;

	size_t find_last_not_of ( const self& str, size_t pos = 0 ) const;
	size_t find_last_not_of ( const T* s, size_t pos, size_t n ) const;
	size_t find_last_not_of ( const T* s, size_t pos = 0 ) const;
	size_t find_last_not_of ( T c, size_t pos = 0 ) const;

	self substr ( size_t pos = 0, size_t n = npos ) const;

	int compare ( const self& str ) const;
	int compare ( const T* s ) const;
	int compare ( size_t pos1, size_t n1, const self& str ) const;
	int compare ( size_t pos1, size_t n1, const T* s) const;
	int compare ( size_t pos1, size_t n1, const self& str, size_t pos2, size_t n2 ) const;
	int compare ( size_t pos1, size_t n1, const T* s, size_t n2) const;

	self& operator+= ( const self& str );
	self& operator+= ( const T* s );
	self& operator+= ( T c );

	self& append ( const self& str );
	self& append ( const self& str, size_t pos, size_t n );
	self& append ( const T* s, size_t n );
	self& append ( const T* s );
	self& append ( size_t n, T c );
	self& append ( T c );
	self& append(const_iterator begin, const_iterator end);
	self& append(const_reverse_iterator begin, const_reverse_iterator end);

	void push_back ( T c );

	self& assign ( const self& str );
	self& assign ( const self& str, size_t pos, size_t n );
	self& assign ( const T* s, size_t n );
	self& assign ( const T* s );
	self& assign ( size_t n, T c );
	self& assign (const_iterator begin, const_iterator end);
	self& assign (const_reverse_iterator begin, const_reverse_iterator end);

	self& insert ( size_t pos1, const self& str );
	self& insert ( size_t pos1, const self& str, size_t pos2, size_t n );
	self& insert ( size_t pos1, const T* s, size_t n);
	self& insert ( size_t pos1, const T* s );
	self& insert ( size_t pos1, size_t n, T c );
	iterator insert ( iterator p, T c );
	void insert ( iterator p, size_t n, T c );
	void insert (iterator p, const_iterator	first, const_iterator last);
	void insert (iterator p, const_reverse_iterator	first, const_reverse_iterator last);

	self& erase ( size_t pos = 0, size_t n = npos );
	iterator erase ( iterator position );
	iterator erase ( iterator first, iterator last );

	self& replace ( size_t pos1, size_t n1,   const self& str );
	self& replace ( iterator i1, iterator i2, const self& str );

	self& replace ( size_t pos1, size_t n1, const self& str, size_t pos2, size_t n2 );

	self& replace ( size_t pos1, size_t n1,   const T* s, size_t n2 );
	self& replace ( iterator i1, iterator i2, const T* s, size_t n2 );

	self& replace ( size_t pos1, size_t n1,   const T* s );
	self& replace ( iterator i1, iterator i2, const T* s );

	self& replace ( size_t pos1, size_t n1,   size_t n2, T c );
	self& replace ( iterator i1, iterator i2, size_t n2, T c );

	self& replace(iterator from, iterator to, iterator src_from, iterator src_end);
	self& replace(iterator from, iterator to, reverse_iterator src_from, reverse_iterator src_end);

	self& replace(const self& what, const self& s);

	void swap ( self& str );

	self& reverse(void);

	self operator + (const self& s) const;
	self operator + (const T *s) const;
	self operator + (T c) const;
	self& trim_left(void);
	self& trim_right(void);
	self& trim(void);

	self& to_lower_case(void);
	self& to_upper_case(void);

	bool equals(const self& s) const;
	bool equals_case_insensitive(const self& s) const;
	bool equals(const T *s) const;
	bool equals_case_insensitive(const T *s) const;

	bool operator <(const self& s) const;
	bool operator >(const self& s) const;
	bool operator <=(const self& s) const;
	bool operator >=(const self& s) const;
	bool operator ==(const self& s) const;
	bool operator ==(const T *s) const;
	bool operator !=(const self& s) const;
	bool operator !=(const T *s) const;

	operator const T*(void) const;

	self align_center(size_t fieldSize) const;
	self align_right(size_t fieldSize) const;
	self align_left(size_t fieldSize) const;
private:
	void init_buffer(void);
	void free_buffer(void);
	void copy(const self& s);
	void copy(const char_type * const s, size_t length);
	void copy(const char_type * const s, size_t length, size_t reserve);
	void copy(const char_type * const s);
	size_t find(const char_type * pattern, size_t patternSize, size_t startPos, size_t searchLen, bool entireMatch, bool findLast, bool outOfSet = false) const;
	int compare(const char_type *what, size_t what_size, const char_type *to_whom, size_t to_whom_size) const;
	bool is_one_of(char_type what, const char_type *pattern, size_t pattern_size) const;

	char_type *_buf;	// our internal buffer containing the string
	size_t _size;		// buffer size
	size_t _str_len;	// actual string length
};

typedef AXISSYSTEMBASE_API AxisString<char_type> String;
typedef AXISSYSTEMBASE_API AxisString<char> AsciiString;
typedef AXISSYSTEMBASE_API AxisString<wchar_t> UnicodeString;

class AXISSYSTEMBASE_API StringEncoding
{
public:
	static char_type * ASCIIToUnicode(const char * s);
	static char * UnicodeToASCII(const char_type * s);
	static void AssignFromASCII(const char * s, String& out);
};

struct AXISSYSTEMBASE_API StringCompareLessThan
{
	bool operator()(const String& a, const String& b) const;
};

struct AXISSYSTEMBASE_API StringCompareGreaterThan
{
	bool operator()(const String& a, const String& b) const;
};

struct AXISSYSTEMBASE_API StringCompareLessOrEqual
{
	bool operator()(const String& a, const String& b) const;
};

struct AXISSYSTEMBASE_API StringCompareGreaterOrEqual
{
	bool operator()(const String& a, const String& b) const;
};

struct AXISSYSTEMBASE_API StringCompareEqual
{
	bool operator()(const String& a, const String& b) const;
};

class AXISSYSTEMBASE_API StringServices
{
public:
	static void Trim(String& out, String& in);
};

AXISSYSTEMBASE_API uint64 hash_value(AsciiString const& str);
AXISSYSTEMBASE_API uint64 hash_value(UnicodeString const& str);

} // namespace axis


AXISSYSTEMBASE_API axis::UnicodeString operator +(const wchar_t *s1, const axis::UnicodeString& s2);
AXISSYSTEMBASE_API axis::UnicodeString operator +(wchar_t s1, const axis::UnicodeString& s2);

AXISSYSTEMBASE_API axis::AsciiString operator +(const char *s1, const axis::AsciiString& s2);
AXISSYSTEMBASE_API axis::AsciiString operator +(char s1, const axis::AsciiString& s2);

