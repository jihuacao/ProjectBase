/****************************************************************************
<one line to give the program's name and a brief idea of what it does.>
Copyright (C) <2021>  <jihuacao>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*****************************************************************************/
#include <ProjectBase/number_theory/find_the_prime_number.hpp>

namespace ProjectBase{
    namespace NumberTheory{
        class PROJECT_BASE_NUMBER_THEORY_SYMBOL SieveOfEratosthenes: FindThePrimeNumber{
            public:
                SieveOfEratosthenes(char* start_ptr, size_t start_size, char* end_ptr, size_t end_size);
                ~SieveOfEratosthenes();
        };
    };
}