#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <filesystem>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <random>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;

class CAFFE_EXPORT TemporaryDirectory
{
public:
  TemporaryDirectory()
  {
    const auto model = std::filesystem::temp_directory_path();
    for (int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++) {
      const auto dir = model / ("caffe_test-" + std::to_string(rd_()));
      if (std::filesystem::create_directory(dir)) {
        path_ = dir;
      }
    }
    LOG(FATAL) << "Failed to create a temporary directory.";
  }

  ~TemporaryDirectory()
  {
    if (!path_.empty())
	  std::filesystem::remove_all(path_);
  }

  [[nodiscard]] const std::filesystem::path& get_path() const
  {
    return path_;
  }

  [[nodiscard]] std::filesystem::path get_temp_filename() const
  {
    if (!path_.empty())
      return path_ / caffe::format_int(static_cast<int>(next_temp_file_++), 9);
    return {};
  }

private:

    std::filesystem::path path_;
    inline static std::random_device rd_{};
    mutable uint64_t next_temp_file_ = 0;
};

CAFFE_EXPORT bool ReadProtoFromTextFile(const char* filename, Message* proto);

CAFFE_EXPORT inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

CAFFE_EXPORT inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

CAFFE_EXPORT inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

CAFFE_EXPORT void WriteProtoToTextFile(const Message& proto, const char* filename);
CAFFE_EXPORT inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

CAFFE_EXPORT bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

CAFFE_EXPORT inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

CAFFE_EXPORT inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

CAFFE_EXPORT inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


CAFFE_EXPORT void WriteProtoToBinaryFile(const Message& proto, const char* filename);
CAFFE_EXPORT inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

CAFFE_EXPORT bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

CAFFE_EXPORT inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

CAFFE_EXPORT bool DecodeDatumNative(Datum* datum);
CAFFE_EXPORT bool DecodeDatum(Datum* datum, bool is_color);

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);


CAFFE_EXPORT bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

CAFFE_EXPORT inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

CAFFE_EXPORT inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

CAFFE_EXPORT inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

CAFFE_EXPORT inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

CAFFE_EXPORT inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}
#endif  // USE_OPENCV

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
