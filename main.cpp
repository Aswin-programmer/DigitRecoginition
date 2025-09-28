#include <iostream>
#include <string>

#include <CSVLoader.h>
#include <NumCPP.h>

int main(int, char**){

    CSVLoader csv = CSVLoader();
    csv.FromCSV((std::string(RESOURCES_PATH) + "customers.csv"), ',');
    std::vector<std::string> first = csv.GetRow(0);
    for(auto it : first)
    {
        std::cout<<it<<" ";
    }
    std::cout<<std::endl;

    // Example 1: broadcasting add
    Tensor<double> a({2,3}, std::vector<double>{1,2,3,4,5,6}); // shape (2,3)
    Tensor<double> b({3}, std::vector<double>{10,20,30});      // shape (3) -> broadcasts to (2,3)
    auto c = a + b;
    std::cout << c.ToString();

    // Example 2: scalar broadcast
    Tensor<double> s({1}, std::vector<double>{5.0});
    auto d = a * s; // multiply all elements by 5
    std::cout << d.ToString();

    // Example 3: vector dot
    Tensor<double> v1({3}, std::vector<double>{1,2,3});
    Tensor<double> v2({3}, std::vector<double>{4,5,6});
    auto dot12 = v1.dot(v2);
    std::cout << "v1·v2 = " << dot12.ToString();

    // Example 4: matrix multiply
    Tensor<double> A({2,3}, std::vector<double>{1,2,3,4,5,6}); // 2x3
    Tensor<double> B({3,2}, std::vector<double>{7,8,9,10,11,12}); // 3x2
    auto M = A.dot(B);
    std::cout << "A·B =\n" << M.ToString();

    std::cout << "Hello, from DigitRecoginition!\n";
}
