#include <string_view>

int main()
{
    constexpr std::string_view name = "vector-experiments";
    return name.empty() ? 1 : 0;
}
