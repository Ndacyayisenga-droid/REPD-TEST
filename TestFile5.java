public class TestFile5 {
    public static void main(String[] args) {
        TestFile5 processor = new TestFile5();
        System.out.println("Launching advanced data structure performance tests...");
        processor.runTests();
        System.out.println("Performance testing phase completed with success.");
    }

    public void runAllTests() {
        testComplexDataStructures();
        testAdvancedAlgorithms();
        testMathematicalComputation();
        testPatternMatching();
        testOptimizationProblems();
    }

    public void testComplexDataStructures() {
        // Binary Search Tree implementation
        BST bst = new BST();
        int[] values = {50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45};
        
        for (int val : values) {
            bst.insert(val);
        }
        
        System.out.println("BST Inorder traversal:");
        bst.inorder();
        System.out.println();
        
        System.out.println("BST search 25: " + bst.search(25));
        System.out.println("BST search 55: " + bst.search(55));
        
        // Trie implementation
        Trie trie = new Trie();
        String[] words = {"apple", "app", "application", "apply", "banana", "band", "bandana"};
        
        for (String word : words) {
            trie.insert(word);
        }
        
        System.out.println("Trie search 'app': " + trie.search("app"));
        System.out.println("Trie prefix 'app': " + trie.startsWith("app"));
        System.out.println("Trie prefix 'ban': " + trie.startsWith("ban"));
        
        // Hash Table with collision handling
        HashTable hashTable = new HashTable(10);
        hashTable.put("key1", "value1");
        hashTable.put("key2", "value2");
        hashTable.put("key11", "value11"); // collision with key1
        
        System.out.println("HashTable get 'key1': " + hashTable.get("key1"));
        System.out.println("HashTable get 'key11': " + hashTable.get("key11"));
        
        // Heap implementation
        MinHeap minHeap = new MinHeap(15);
        int[] heapValues = {40, 100, 20, 25, 16, 30, 5, 70, 90, 85};
        
        for (int val : heapValues) {
            minHeap.insert(val);
        }
        
        System.out.println("Min heap elements:");
        while (!minHeap.isEmpty()) {
            System.out.print(minHeap.extractMin() + " ");
        }
        System.out.println();
    }

    public void testAdvancedAlgorithms() {
        // Dynamic Programming - Longest Common Subsequence
        String str1 = "ABCDGH";
        String str2 = "AEDFHR";
        int lcs = longestCommonSubsequence(str1, str2);
        System.out.println("LCS of '" + str1 + "' and '" + str2 + "': " + lcs);
        
        // Knapsack problem
        int[] weights = {10, 20, 30};
        int[] values = {60, 100, 120};
        int capacity = 50;
        int maxValue = knapsack(weights, values, capacity);
        System.out.println("Knapsack max value: " + maxValue);
        
        // Graph algorithms - Dijkstra's shortest path
        int[][] graph = {
            {0, 4, 0, 0, 0, 0, 0, 8, 0},
            {4, 0, 8, 0, 0, 0, 0, 11, 0},
            {0, 8, 0, 7, 0, 4, 0, 0, 2},
            {0, 0, 7, 0, 9, 14, 0, 0, 0},
            {0, 0, 0, 9, 0, 10, 0, 0, 0},
            {0, 0, 4, 14, 10, 0, 2, 0, 0},
            {0, 0, 0, 0, 0, 2, 0, 1, 6},
            {8, 11, 0, 0, 0, 0, 1, 0, 7},
            {0, 0, 2, 0, 0, 0, 6, 7, 0}
        };
        
        int[] distances = dijkstra(graph, 0);
        System.out.println("Shortest distances from vertex 0:");
        for (int i = 0; i < distances.length; i++) {
            System.out.println("To vertex " + i + ": " + distances[i]);
        }
        
        // Quick select for finding kth smallest element
        int[] arr = {3, 2, 1, 5, 6, 4};
        int k = 2;
        int kthSmallest = quickSelect(arr.clone(), 0, arr.length - 1, k - 1);
        System.out.println(k + "th smallest element: " + kthSmallest);
    }

    public void testMathematicalComputation() {
        // Matrix operations
        int[][] matrix1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        int[][] matrix2 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        
        int[][] product = matrixMultiply(matrix1, matrix2);
        System.out.println("Matrix multiplication result:");
        printMatrix(product);
        
        // Determinant calculation
        double det = determinant(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}});
        System.out.println("Matrix determinant: " + det);
        
        // Fast exponentiation
        long result = fastPower(2, 20);
        System.out.println("2^20 = " + result);
        
        // Prime factorization
        int number = 315;
        java.util.List<Integer> factors = primeFactorization(number);
        System.out.println("Prime factors of " + number + ": " + factors);
        
        // GCD and LCM for multiple numbers
        int[] numbers = {48, 18, 24, 12};
        int gcdResult = gcdArray(numbers);
        int lcmResult = lcmArray(numbers);
        System.out.println("GCD of array: " + gcdResult);
        System.out.println("LCM of array: " + lcmResult);
        
        // Fibonacci sequence with memoization
        java.util.Map<Integer, Long> fibMemo = new java.util.HashMap<>();
        for (int i = 0; i <= 50; i++) {
            System.out.println("Fibonacci(" + i + ") = " + fibonacciMemo(i, fibMemo));
        }
    }

    public void testPatternMatching() {
        String text = "ABABDABACDABABCABCABCABCABC";
        String pattern = "ABABCABCABCABC";
        
        // KMP algorithm
        int kmpIndex = kmpSearch(text, pattern);
        System.out.println("KMP search result: " + kmpIndex);
        
        // Rabin-Karp algorithm
        int rkIndex = rabinKarpSearch(text, pattern);
        System.out.println("Rabin-Karp search result: " + rkIndex);
        
        // Boyer-Moore algorithm (simplified)
        int bmIndex = boyerMooreSearch(text, pattern);
        System.out.println("Boyer-Moore search result: " + bmIndex);
        
        // Regular expression matching
        String[] testStrings = {"hello", "helo", "hellooo", "hell"};
        String regex = "hel+o*";
        
        for (String str : testStrings) {
            boolean matches = regexMatch(str, regex);
            System.out.println("'" + str + "' matches '" + regex + "': " + matches);
        }
        
        // Wildcard pattern matching
        String[] patterns = {"h*o", "h?llo", "*llo", "he*"};
        String testStr = "hello";
        
        for (String pat : patterns) {
            boolean matches = wildcardMatch(testStr, pat);
            System.out.println("'" + testStr + "' matches '" + pat + "': " + matches);
        }
    }

    public void testOptimizationProblems() {
        // Traveling Salesman Problem (brute force for small instances)
        int[][] distanceMatrix = {
            {0, 10, 15, 20},
            {10, 0, 35, 25},
            {15, 35, 0, 30},
            {20, 25, 30, 0}
        };
        
        int minCost = travelingSalesman(distanceMatrix);
        System.out.println("TSP minimum cost: " + minCost);
        
        // Job scheduling problem
        Job[] jobs = {
            new Job(1, 4, 20),
            new Job(2, 1, 10),
            new Job(3, 1, 40),
            new Job(4, 1, 30)
        };
        
        int maxProfit = jobScheduling(jobs);
        System.out.println("Job scheduling max profit: " + maxProfit);
        
        // Coin change problem
        int[] coins = {1, 5, 10, 25};
        int amount = 67;
        int minCoins = coinChange(coins, amount);
        System.out.println("Minimum coins for " + amount + ": " + minCoins);
        
        // Edit distance (Levenshtein distance)
        String word1 = "intention";
        String word2 = "execution";
        int editDist = editDistance(word1, word2);
        System.out.println("Edit distance between '" + word1 + "' and '" + word2 + "': " + editDist);
        
        // Maximum subarray sum (Kadane's algorithm)
        int[] array = {-2, -3, 4, -1, -2, 1, 5, -3};
        int maxSum = maxSubarraySum(array);
        System.out.println("Maximum subarray sum: " + maxSum);
    }

    // Data structure implementations
    static class BST {
        class Node {
            int data;
            Node left, right;
            
            Node(int data) {
                this.data = data;
                left = right = null;
            }
        }
        
        Node root;
        
        void insert(int data) {
            root = insertRec(root, data);
        }
        
        Node insertRec(Node root, int data) {
            if (root == null) {
                root = new Node(data);
                return root;
            }
            
            if (data < root.data) {
                root.left = insertRec(root.left, data);
            } else if (data > root.data) {
                root.right = insertRec(root.right, data);
            }
            
            return root;
        }
        
        boolean search(int data) {
            return searchRec(root, data);
        }
        
        boolean searchRec(Node root, int data) {
            if (root == null) return false;
            if (root.data == data) return true;
            
            return data < root.data ? searchRec(root.left, data) : searchRec(root.right, data);
        }
        
        void inorder() {
            inorderRec(root);
        }
        
        void inorderRec(Node root) {
            if (root != null) {
                inorderRec(root.left);
                System.out.print(root.data + " ");
                inorderRec(root.right);
            }
        }
    }
    
    static class Trie {
        class TrieNode {
            TrieNode[] children = new TrieNode[26];
            boolean isEndOfWord = false;
        }
        
        TrieNode root;
        
        Trie() {
            root = new TrieNode();
        }
        
        void insert(String word) {
            TrieNode curr = root;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (curr.children[index] == null) {
                    curr.children[index] = new TrieNode();
                }
                curr = curr.children[index];
            }
            curr.isEndOfWord = true;
        }
        
        boolean search(String word) {
            TrieNode curr = root;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (curr.children[index] == null) {
                    return false;
                }
                curr = curr.children[index];
            }
            return curr.isEndOfWord;
        }
        
        boolean startsWith(String prefix) {
            TrieNode curr = root;
            for (char c : prefix.toCharArray()) {
                int index = c - 'a';
                if (curr.children[index] == null) {
                    return false;
                }
                curr = curr.children[index];
            }
            return true;
        }
    }
    
    static class HashTable {
        class Entry {
            String key;
            String value;
            Entry next;
            
            Entry(String key, String value) {
                this.key = key;
                this.value = value;
            }
        }
        
        Entry[] table;
        int size;
        
        HashTable(int capacity) {
            table = new Entry[capacity];
            size = 0;
        }
        
        int hash(String key) {
            return Math.abs(key.hashCode()) % table.length;
        }
        
        void put(String key, String value) {
            int index = hash(key);
            Entry entry = table[index];
            
            while (entry != null) {
                if (entry.key.equals(key)) {
                    entry.value = value;
                    return;
                }
                entry = entry.next;
            }
            
            Entry newEntry = new Entry(key, value);
            newEntry.next = table[index];
            table[index] = newEntry;
            size++;
        }
        
        String get(String key) {
            int index = hash(key);
            Entry entry = table[index];
            
            while (entry != null) {
                if (entry.key.equals(key)) {
                    return entry.value;
                }
                entry = entry.next;
            }
            
            return null;
        }
    }
    
    static class MinHeap {
        int[] heap;
        int size;
        int capacity;
        
        MinHeap(int capacity) {
            this.capacity = capacity;
            this.size = 0;
            this.heap = new int[capacity];
        }
        
        int parent(int i) { return (i - 1) / 2; }
        int leftChild(int i) { return 2 * i + 1; }
        int rightChild(int i) { return 2 * i + 2; }
        
        void swap(int i, int j) {
            int temp = heap[i];
            heap[i] = heap[j];
            heap[j] = temp;
        }
        
        void insert(int key) {
            if (size == capacity) return;
            
            size++;
            int i = size - 1;
            heap[i] = key;
            
            while (i != 0 && heap[parent(i)] > heap[i]) {
                swap(i, parent(i));
                i = parent(i);
            }
        }
        
        int extractMin() {
            if (size <= 0) return Integer.MAX_VALUE;
            if (size == 1) {
                size--;
                return heap[0];
            }
            
            int root = heap[0];
            heap[0] = heap[size - 1];
            size--;
            minHeapify(0);
            
            return root;
        }
        
        void minHeapify(int i) {
            int left = leftChild(i);
            int right = rightChild(i);
            int smallest = i;
            
            if (left < size && heap[left] < heap[smallest]) {
                smallest = left;
            }
            
            if (right < size && heap[right] < heap[smallest]) {
                smallest = right;
            }
            
            if (smallest != i) {
                swap(i, smallest);
                minHeapify(smallest);
            }
        }
        
        boolean isEmpty() {
            return size == 0;
        }
    }
    
    static class Job {
        int id, deadline, profit;
        
        Job(int id, int deadline, int profit) {
            this.id = id;
            this.deadline = deadline;
            this.profit = profit;
        }
    }

    // Algorithm implementations
    private int longestCommonSubsequence(String str1, String str2) {
        int m = str1.length(), n = str2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }

    private int knapsack(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];
        
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        
        return dp[n][capacity];
    }

    private int[] dijkstra(int[][] graph, int src) {
        int V = graph.length;
        int[] dist = new int[V];
        boolean[] visited = new boolean[V];
        
        java.util.Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;
        
        for (int count = 0; count < V - 1; count++) {
            int u = minDistance(dist, visited);
            visited[u] = true;
            
            for (int v = 0; v < V; v++) {
                if (!visited[v] && graph[u][v] != 0 && dist[u] != Integer.MAX_VALUE
                    && dist[u] + graph[u][v] < dist[v]) {
                    dist[v] = dist[u] + graph[u][v];
                }
            }
        }
        
        return dist;
    }

    private int minDistance(int[] dist, boolean[] visited) {
        int min = Integer.MAX_VALUE, minIndex = -1;
        
        for (int v = 0; v < dist.length; v++) {
            if (!visited[v] && dist[v] <= min) {
                min = dist[v];
                minIndex = v;
            }
        }
        
        return minIndex;
    }

    private int quickSelect(int[] arr, int low, int high, int k) {
        if (low == high) return arr[low];
        
        int pivotIndex = partition(arr, low, high);
        
        if (k == pivotIndex) {
            return arr[k];
        } else if (k < pivotIndex) {
            return quickSelect(arr, low, pivotIndex - 1, k);
        } else {
            return quickSelect(arr, pivotIndex + 1, high, k);
        }
    }

    private int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr, i, j);
            }
        }
        
        swap(arr, i + 1, high);
        return i + 1;
    }

    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // Mathematical computation methods
    private int[][] matrixMultiply(int[][] a, int[][] b) {
        int rows = a.length;
        int cols = b[0].length;
        int common = a[0].length;
        int[][] result = new int[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }

    private void printMatrix(int[][] matrix) {
        for (int[] row : matrix) {
            for (int val : row) {
                System.out.print(val + " ");
            }
            System.out.println();
        }
    }

    private double determinant(double[][] matrix) {
        int n = matrix.length;
        if (n == 1) return matrix[0][0];
        if (n == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        
        double det = 0;
        for (int i = 0; i < n; i++) {
            double[][] subMatrix = new double[n - 1][n - 1];
            for (int j = 1; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (k < i) {
                        subMatrix[j - 1][k] = matrix[j][k];
                    } else if (k > i) {
                        subMatrix[j - 1][k - 1] = matrix[j][k];
                    }
                }
            }
            det += Math.pow(-1, i) * matrix[0][i] * determinant(subMatrix);
        }
        
        return det;
    }

    private long fastPower(long base, int exp) {
        if (exp == 0) return 1;
        if (exp == 1) return base;
        
        if (exp % 2 == 0) {
            long half = fastPower(base, exp / 2);
            return half * half;
        } else {
            return base * fastPower(base, exp - 1);
        }
    }

    private java.util.List<Integer> primeFactorization(int n) {
        java.util.List<Integer> factors = new java.util.ArrayList<>();
        
        for (int i = 2; i <= Math.sqrt(n); i++) {
            while (n % i == 0) {
                factors.add(i);
                n /= i;
            }
        }
        
        if (n > 1) factors.add(n);
        
        return factors;
    }

    private int gcdArray(int[] arr) {
        int result = arr[0];
        for (int i = 1; i < arr.length; i++) {
            result = gcd(result, arr[i]);
        }
        return result;
    }

    private int lcmArray(int[] arr) {
        int result = arr[0];
        for (int i = 1; i < arr.length; i++) {
            result = lcm(result, arr[i]);
        }
        return result;
    }

    private int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    private int lcm(int a, int b) {
        return (a * b) / gcd(a, b);
    }

    private long fibonacciMemo(int n, java.util.Map<Integer, Long> memo) {
        if (n <= 1) return n;
        if (memo.containsKey(n)) return memo.get(n);
        
        long result = fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo);
        memo.put(n, result);
        return result;
    }

    // Pattern matching algorithms
    private int kmpSearch(String text, String pattern) {
        int[] lps = computeLPS(pattern);
        int i = 0, j = 0;
        
        while (i < text.length()) {
            if (pattern.charAt(j) == text.charAt(i)) {
                i++;
                j++;
            }
            
            if (j == pattern.length()) {
                return i - j;
            } else if (i < text.length() && pattern.charAt(j) != text.charAt(i)) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return -1;
    }

    private int[] computeLPS(String pattern) {
        int[] lps = new int[pattern.length()];
        int len = 0;
        int i = 1;
        
        while (i < pattern.length()) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0) {
                    len = lps[len - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        
        return lps;
    }

    private int rabinKarpSearch(String text, String pattern) {
        int d = 256; // number of characters
        int q = 101; // prime number
        int m = pattern.length();
        int n = text.length();
        int p = 0; // hash value for pattern
        int t = 0; // hash value for text
        int h = 1;
        
        for (int i = 0; i < m - 1; i++) {
            h = (h * d) % q;
        }
        
        for (int i = 0; i < m; i++) {
            p = (d * p + pattern.charAt(i)) % q;
            t = (d * t + text.charAt(i)) % q;
        }
        
        for (int i = 0; i <= n - m; i++) {
            if (p == t) {
                boolean match = true;
                for (int j = 0; j < m; j++) {
                    if (text.charAt(i + j) != pattern.charAt(j)) {
                        match = false;
                        break;
                    }
                }
                if (match) return i;
            }
            
            if (i < n - m) {
                t = (d * (t - text.charAt(i) * h) + text.charAt(i + m)) % q;
                if (t < 0) t = t + q;
            }
        }
        
        return -1;
    }

    private int boyerMooreSearch(String text, String pattern) {
        // Simplified Boyer-Moore (bad character heuristic only)
        int[] badChar = new int[256];
        java.util.Arrays.fill(badChar, -1);
        
        for (int i = 0; i < pattern.length(); i++) {
            badChar[pattern.charAt(i)] = i;
        }
        
        int s = 0;
        while (s <= text.length() - pattern.length()) {
            int j = pattern.length() - 1;
            
            while (j >= 0 && pattern.charAt(j) == text.charAt(s + j)) {
                j--;
            }
            
            if (j < 0) {
                return s;
            } else {
                s += Math.max(1, j - badChar[text.charAt(s + j)]);
            }
        }
        
        return -1;
    }

    private boolean regexMatch(String str, String pattern) {
        // Simple regex matcher (supports . and *)
        return isMatch(str, pattern, 0, 0);
    }

    private boolean isMatch(String str, String pattern, int i, int j) {
        if (j == pattern.length()) return i == str.length();
        
        boolean firstMatch = i < str.length() && 
                           (pattern.charAt(j) == str.charAt(i) || pattern.charAt(j) == '.');
        
        if (j + 1 < pattern.length() && pattern.charAt(j + 1) == '*') {
            return isMatch(str, pattern, i, j + 2) || 
                   (firstMatch && isMatch(str, pattern, i + 1, j));
        } else {
            return firstMatch && isMatch(str, pattern, i + 1, j + 1);
        }
    }

    private boolean wildcardMatch(String str, String pattern) {
        int m = str.length();
        int n = pattern.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        
        dp[0][0] = true;
        
        for (int j = 1; j <= n; j++) {
            if (pattern.charAt(j - 1) == '*') {
                dp[0][j] = dp[0][j - 1];
            }
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (pattern.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                } else if (pattern.charAt(j - 1) == '?' || 
                          str.charAt(i - 1) == pattern.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        
        return dp[m][n];
    }

    // Optimization problems
    private int travelingSalesman(int[][] dist) {
        int n = dist.length;
        int[] cities = new int[n - 1];
        for (int i = 1; i < n; i++) {
            cities[i - 1] = i;
        }
        
        int minCost = Integer.MAX_VALUE;
        
        do {
            int currentCost = 0;
            int current = 0;
            
            for (int city : cities) {
                currentCost += dist[current][city];
                current = city;
            }
            currentCost += dist[current][0];
            
            minCost = Math.min(minCost, currentCost);
        } while (nextPermutation(cities));
        
        return minCost;
    }

    private boolean nextPermutation(int[] arr) {
        int i = arr.length - 2;
        while (i >= 0 && arr[i] >= arr[i + 1]) i--;
        
        if (i < 0) return false;
        
        int j = arr.length - 1;
        while (arr[j] <= arr[i]) j--;
        
        swap(arr, i, j);
        reverse(arr, i + 1);
        
        return true;
    }

    private void reverse(int[] arr, int start) {
        int end = arr.length - 1;
        while (start < end) {
            swap(arr, start++, end--);
        }
    }

    private int jobScheduling(Job[] jobs) {
        java.util.Arrays.sort(jobs, (a, b) -> Integer.compare(b.profit, a.profit));
        
        int maxDeadline = 0;
        for (Job job : jobs) {
            maxDeadline = Math.max(maxDeadline, job.deadline);
        }
        
        boolean[] slot = new boolean[maxDeadline];
        int totalProfit = 0;
        
        for (Job job : jobs) {
            for (int j = Math.min(maxDeadline, job.deadline) - 1; j >= 0; j--) {
                if (!slot[j]) {
                    slot[j] = true;
                    totalProfit += job.profit;
                    break;
                }
            }
        }
        
        return totalProfit;
    }

    private int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        java.util.Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        
        return dp[amount] > amount ? -1 : dp[amount];
    }

    private int editDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }

    private int maxSubarraySum(int[] arr) {
        int maxSoFar = arr[0];
        int maxEndingHere = arr[0];
        
        for (int i = 1; i < arr.length; i++) {
            maxEndingHere = Math.max(arr[i], maxEndingHere + arr[i]);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        
        return maxSoFar;
    }
}
