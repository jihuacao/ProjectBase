/**
 * @brief
 * @note
 * */

template<typename type> void quick_sort(type arr[], unsigned long long from, unsigned long long end){
    int povit = arr[from];
    int l = from;
    int r = end;
    int len = end + 1;
    while(l <= r)
    {
        while(l < end && arr[l] <= povit) l++;
        while(r > from && arr[r] >= povit) r--;
        if(r < len && l > 0 && r > l)
        {
            int temp = arr[l];
            arr[l] = arr[r];
            arr[r] = temp;
        }
    }

    arr[from] = arr[r];
    arr[r] = povit;

    quick_sort(arr, from, r - 1);
    quick_sort(arr, r + 1, end);
};


//template<typename type> void a(type c){};