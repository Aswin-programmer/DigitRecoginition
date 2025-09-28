#include "CSVLoader.h"

CSVLoader::CSVLoader()
{
}

CSVLoader::~CSVLoader()
{
}

std::vector<std::vector<std::string>> CSVLoader::ParseCSVFile(const std::string &path, char delim)
{
    std::ifstream ifs(path, std::ios::binary);
    if(!ifs)
    {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;

    for(size_t i=0;i<content.size();i++)
    {
        char c = content[i];

        if(c == '"')
        {
            //If in quotes and next char is also a quote -> escape
            if(in_quotes && i+1 < content.size() && content[i+1] == '"')
            {
                field.push_back('"');
                ++i;
            }
            else
            {
                in_quotes = !in_quotes;
            }
        }
        else if(c == delim && !in_quotes)
        {
            row.push_back(field);
            field.clear();
        }
        else if((c == '\n' || c == '\r') && !in_quotes)
        {
            if(c == '\r' && i+1 < content.size() && content[i+1] == '\n')
            {
                ++i;
            }
            row.push_back(field);
            field.clear();
            rows.push_back(row);
            row.clear();
        }
        else
        {
            field.push_back(c);
        }
    }

    //Push final field/row if necessary
    if(!in_quotes)
    {
        //final field:
        row.push_back(field);
        if(!row.empty())
        {
            rows.push_back(row);
        }
    }
    else
    {
        throw std::runtime_error("Unterminiated quoted field in CSV!.");
    }

    return rows;
}

bool CSVLoader::FromCSV(const std::string &path, char delim, bool header)
{
    std::vector<std::vector<std::string>> rows = ParseCSVFile(path, delim);
    if(rows.empty())
    {
        return false;
    }

    size_t header_row = 0;
    std::vector<std::string> names;
    if(header)
    {
        names = rows[0];
        header_row = 1;
    }
    else
    {
        size_t ncols = rows[0].size();
        names.resize(ncols);
        for(size_t i=0;i<ncols;i++)
        {
            names[i] = "col" + std::to_string(i);
            header_row = 0;
        }
    }

    size_t ncols = names.size();
    size_t nrows = (rows.size() > header_row) ? rows.size() - header_row : 0;
    std::vector<std::vector<std::string>> columns(ncols, std::vector<std::string>());
    for(size_t c=0;c<ncols;c++)
    {
        columns.reserve(nrows);
    }

    for(size_t r = header_row;r<rows.size();r++)
    {
        std::vector<std::string>& row = rows[r];
        // pad if row is shorter
        row.resize(ncols);
        for(size_t c = 0;c<ncols;c++)
        {
            columns[c].push_back(row[c]);
        }
    }

    col_names = std::move(names);
    cols = std::move(columns);
    InferTypes();
    return true;
}

std::vector<std::string> CSVLoader::GetRow(size_t r) const
{
    std::vector<std::string> row;
    row.reserve(NCols());
    for(size_t c=0;c<NCols();c++)
    {
        row.push_back(cols[c][r]);
    }
    return row;
}

void CSVLoader::InferTypes(size_t sample_rows)
{
    dtypes.assign(NCols(), DType::String);
    size_t rows_to_sample = std::min(sample_rows, NRows());
    for(size_t c=0;c<NCols();c++)
    {
        bool all_int = true;
        bool all_float = true;
        bool all_bool = true;
        for(size_t r=0;r<rows_to_sample;r++)
        {
            const std::string& s = cols[c][r];
            if(s.empty()) continue;
            if(all_int && !LooksLikeInt(s)) all_int=false;
            if(all_float && !LooksLikeFloat(s)) all_float=false;
            if(all_bool && !LooksLikeBool(s)) all_bool=false;
        }
        if(all_int) dtypes[c] = DType::Int64;
        else if(all_float) dtypes[c] = DType::Double;
        else if(all_bool) dtypes[c] = DType::Bool;
        else dtypes[c] = DType::String;
    }
}

bool CSVLoader::LooksLikeInt(const std::string &s)
{
    if(s.empty())
    {
        return false;
    }

    size_t i = 0;
    if(s[0] == '+' || s[0] == '-')
    {
        i = 1;
    }

    if(i == s.size())
    {
        return false;
    }

    for(; i<s.size();i++)
    {
        if(!std::isdigit(static_cast<unsigned char>(s[i])))
        {
            return false;
        }
    }

    return true;
}

bool CSVLoader::LooksLikeFloat(const std::string &s)
{
    if(s.empty())
    {
        return true;
    }

    char* endptr = nullptr;
    errno = 0;
    const char* c = s.c_str();
    std::strtod(c, &endptr);
    if(endptr == c) return false;
    if(*endptr != '\0') return false;
    if(errno == ERANGE) return false;
    
    return true;
}

bool CSVLoader::LooksLikeBool(const std::string &s)
{
    std::string t;
    t.reserve(s.size());
    for(char c : s)
    {
        t.push_back(std::tolower(static_cast<unsigned char>(c)));
    }
    return (t=="true" || t=="false" || t=="1" || t=="0");
}
