#ifndef CAFFE_UTIL_DB_HPP
#define CAFFE_UTIL_DB_HPP

#include <string>

#include "caffe/common.hpp"


namespace caffe {

enum DataParameter_DB : int;

	namespace db {

enum Mode { READ, WRITE, NEW };

class CAFFE_EXPORT Cursor {
 public:
  Cursor() = default;
  virtual ~Cursor() = default;
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual std::string key() = 0;
  virtual std::string value() = 0;
  virtual bool valid() = 0;

  DISABLE_COPY_AND_ASSIGN(Cursor);
};

class CAFFE_EXPORT Transaction {
 public:
  Transaction() = default;
  virtual ~Transaction() = default;
  virtual void Put(const std::string& key, const std::string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

class CAFFE_EXPORT DB {
 public:
  DB() = default;
  virtual ~DB() = default;
  virtual void Open(const std::string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;

  DISABLE_COPY_AND_ASSIGN(DB);
};

CAFFE_EXPORT DB* GetDB(DataParameter_DB backend);
CAFFE_EXPORT DB* GetDB(const std::string& backend);

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_HPP
