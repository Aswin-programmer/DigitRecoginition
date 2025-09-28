#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <functional>
#include <cerrno>
#include <cstdlib>
#include <limits>

enum class DType
{
    String,
    Int64,
    Double,
    Bool
};

class CSVLoader
{
    public:
    CSVLoader();
    ~CSVLoader();

    size_t NRows() const {return cols.empty() ? 0 : cols[0].size();}
    size_t NCols() const {return cols.size();}
    bool FromCSV(const std::string& path, char delim=',', bool header=true);
    std::vector<std::string> GetRow(size_t r) const;

    public:
    
    private:
    std::vector<std::string> col_names;
    std::vector<std::vector<std::string>> cols;
    std::vector<DType> dtypes;

    private:
    std::vector<std::vector<std::string>> ParseCSVFile(const std::string& path
    , char delim = ',');
    void InferTypes(size_t sample_rows = 1000);
    bool LooksLikeInt(const std::string& s);
    bool LooksLikeFloat(const std::string& s);
    bool LooksLikeBool(const std::string& s);
};

