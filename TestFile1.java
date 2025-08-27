package test;

import java.util.*;
import java.util.stream.*;
import java.nio.file.*;
import java.time.*;
import java.util.concurrent.*;
import java.io.*;

public class TestFile1 {
    private String name;
    private int id;
    private List<String> items;
    private Map<String, Object> properties;

    public TestFile1(String name, int id) {
        this.name = name;
        this.id = id;
        this.items = new ArrayList<>();
        this.properties = new HashMap<>();
    }

    public static void main(String[] args) {
        TestFile1 test = new TestFile1("TestInstance", 1);
        test.runTests();
    }

    public void runTests() {
        // Data structure tests
        testArrayOperations();
        testListOperations();
        testMapOperations();
        testSetOperations();
        testQueueOperations();
        
        // Algorithm tests
        testSortingAlgorithms();
        testSearchAlgorithms();
        testGraphAlgorithms();
        testDynamicProgramming();
        
        // Concurrent programming tests
        testThreading();
        testSynchronization();
        testCompletableFuture();
        
        // File I/O tests
        testFileOperations();
        testStreamProcessing();
        
        // Error handling tests
        testExceptionHandling();
        testResourceManagement();
    }

    public void testArrayOperations() {
        int[] numbers = {5, 2, 8, 1, 9, 3};
        System.out.println("Original array: " + Arrays.toString(numbers));
        
        // Bubble sort implementation
        bubbleSort(numbers);
        System.out.println("Bubble sorted: " + Arrays.toString(numbers));
        
        // Binary search
        int target = 8;
        int index = binarySearch(numbers, target);
        System.out.println("Index of " + target + ": " + index);
        
        // Array manipulation
        int[] doubled = doubleArray(numbers);
        System.out.println("Doubled: " + Arrays.toString(doubled));
        
        // Matrix operations
        int[][] matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        int[][] transposed = transposeMatrix(matrix);
        System.out.println("Transposed matrix: " + Arrays.deepToString(transposed));
    }

    private void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }

    private int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }

    private int[] doubleArray(int[] arr) {
        return Arrays.stream(arr).map(x -> x * 2).toArray();
    }

    private int[][] transposeMatrix(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] result = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public void testListOperations() {
        List<String> fruits = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");
        System.out.println("Original fruits: " + fruits);
        
        // Filter and transform
        List<String> upperFruits = fruits.stream()
            .filter(f -> f.length() > 5)
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Filtered and uppercase: " + upperFruits);
        
        // Custom sorting
        List<String> sortedByLength = fruits.stream()
            .sorted(Comparator.comparing(String::length).thenComparing(String::compareTo))
            .collect(Collectors.toList());
        System.out.println("Sorted by length: " + sortedByLength);
        
        // Grouping
        Map<Integer, List<String>> groupedByLength = fruits.stream()
            .collect(Collectors.groupingBy(String::length));
        System.out.println("Grouped by length: " + groupedByLength);
    }

    public void testMapOperations() {
        Map<String, Integer> wordCounts = new HashMap<>();
        String text = "hello world hello java world programming";
        String[] words = text.split(" ");
        
        // Count word frequencies
        for (String word : words) {
            wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
        }
        System.out.println("Word counts: " + wordCounts);
        
        // Find most frequent word
        String mostFrequent = wordCounts.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("");
        System.out.println("Most frequent word: " + mostFrequent);
        
        // Merge maps
        Map<String, Integer> additionalCounts = Map.of("java", 5, "programming", 3);
        additionalCounts.forEach((k, v) -> wordCounts.merge(k, v, Integer::sum));
        System.out.println("After merge: " + wordCounts);
    }

    public void testSetOperations() {
        Set<Integer> set1 = Set.of(1, 2, 3, 4, 5);
        Set<Integer> set2 = Set.of(4, 5, 6, 7, 8);
        
        // Union
        Set<Integer> union = new HashSet<>(set1);
        union.addAll(set2);
        System.out.println("Union: " + union);
        
        // Intersection
        Set<Integer> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        System.out.println("Intersection: " + intersection);
        
        // Difference
        Set<Integer> difference = new HashSet<>(set1);
        difference.removeAll(set2);
        System.out.println("Difference: " + difference);
    }

    public void testQueueOperations() {
        Queue<String> queue = new LinkedList<>();
        
        // Add elements
        queue.offer("first");
        queue.offer("second");
        queue.offer("third");
        System.out.println("Queue: " + queue);
        
        // Priority queue
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(Collections.reverseOrder());
        priorityQueue.addAll(Arrays.asList(3, 1, 4, 1, 5, 9, 2, 6));
        
        System.out.print("Priority queue (descending): ");
        while (!priorityQueue.isEmpty()) {
            System.out.print(priorityQueue.poll() + " ");
        }
        System.out.println();
    }

    public void testSortingAlgorithms() {
        int[] data = {64, 34, 25, 12, 22, 11, 90};
        System.out.println("Original: " + Arrays.toString(data));
        
        // Quick sort
        int[] quickSorted = data.clone();
        quickSort(quickSorted, 0, quickSorted.length - 1);
        System.out.println("Quick sorted: " + Arrays.toString(quickSorted));
        
        // Merge sort
        int[] mergeSorted = data.clone();
        mergeSort(mergeSorted, 0, mergeSorted.length - 1);
        System.out.println("Merge sorted: " + Arrays.toString(mergeSorted));
        
        // Heap sort
        int[] heapSorted = data.clone();
        heapSort(heapSorted);
        System.out.println("Heap sorted: " + Arrays.toString(heapSorted));
    }

    private void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pivot = partition(arr, low, high);
            quickSort(arr, low, pivot - 1);
            quickSort(arr, pivot + 1, high);
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

    private void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }

    private void merge(int[] arr, int left, int mid, int right) {
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
            }
        }
        
        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];
        
        System.arraycopy(temp, 0, arr, left, temp.length);
    }

    private void heapSort(int[] arr) {
        int n = arr.length;
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
        for (int i = n - 1; i > 0; i--) {
            swap(arr, 0, i);
            heapify(arr, i, 0);
        }
    }

    private void heapify(int[] arr, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) largest = left;
        if (right < n && arr[right] > arr[largest]) largest = right;
        
        if (largest != i) {
            swap(arr, i, largest);
            heapify(arr, n, largest);
        }
    }

    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    public void testSearchAlgorithms() {
        String text = "ABABDABACDABABCABCABCABCABC";
        String pattern = "ABABCABCABCABC";
        
        int index = kmpSearch(text, pattern);
        System.out.println("KMP search result: " + index);
        
        // Linear search in list
        List<Integer> numbers = Arrays.asList(5, 2, 8, 1, 9, 3, 7, 4, 6);
        int target = 7;
        int linearIndex = linearSearch(numbers, target);
        System.out.println("Linear search for " + target + ": " + linearIndex);
    }

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

    private int linearSearch(List<Integer> list, int target) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == target) {
                return i;
            }
        }
        return -1;
    }

    public void testGraphAlgorithms() {
        // Simple graph representation
        Map<Integer, List<Integer>> graph = new HashMap<>();
        graph.put(0, Arrays.asList(1, 2));
        graph.put(1, Arrays.asList(0, 3, 4));
        graph.put(2, Arrays.asList(0, 5, 6));
        graph.put(3, Arrays.asList(1));
        graph.put(4, Arrays.asList(1));
        graph.put(5, Arrays.asList(2));
        graph.put(6, Arrays.asList(2));
        
        System.out.println("DFS traversal:");
        dfs(graph, 0, new HashSet<>());
        System.out.println();
        
        System.out.println("BFS traversal:");
        bfs(graph, 0);
    }

    private void dfs(Map<Integer, List<Integer>> graph, int node, Set<Integer> visited) {
        visited.add(node);
        System.out.print(node + " ");
        
        for (int neighbor : graph.getOrDefault(node, Collections.emptyList())) {
            if (!visited.contains(neighbor)) {
                dfs(graph, neighbor, visited);
            }
        }
    }

    private void bfs(Map<Integer, List<Integer>> graph, int start) {
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        
        queue.offer(start);
        visited.add(start);
        
        while (!queue.isEmpty()) {
            int node = queue.poll();
            System.out.print(node + " ");
            
            for (int neighbor : graph.getOrDefault(node, Collections.emptyList())) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        System.out.println();
    }

    public void testDynamicProgramming() {
        // Fibonacci with memoization
        Map<Integer, Long> fibMemo = new HashMap<>();
        System.out.println("Fibonacci(50): " + fibonacciMemo(50, fibMemo));
        
        // Longest common subsequence
        String str1 = "ABCDGH";
        String str2 = "AEDFHR";
        int lcs = longestCommonSubsequence(str1, str2);
        System.out.println("LCS of '" + str1 + "' and '" + str2 + "': " + lcs);
        
        // Coin change problem
        int[] coins = {1, 5, 10, 25};
        int amount = 30;
        int ways = coinChange(coins, amount);
        System.out.println("Ways to make " + amount + " cents: " + ways);
    }

    private long fibonacciMemo(int n, Map<Integer, Long> memo) {
        if (n <= 1) return n;
        if (memo.containsKey(n)) return memo.get(n);
        
        long result = fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo);
        memo.put(n, result);
        return result;
    }

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

    private int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }

    public void testThreading() {
        System.out.println("Testing threading operations...");
        
        // Thread creation and execution
        Thread thread1 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Thread 1: " + i);
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });
        
        Thread thread2 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Thread 2: " + i);
                try {
                    Thread.sleep(150);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });
        
        thread1.start();
        thread2.start();
        
        try {
            thread1.join();
            thread2.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public void testSynchronization() {
        Counter counter = new Counter();
        
        Runnable incrementTask = () -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        };
        
        Thread t1 = new Thread(incrementTask);
        Thread t2 = new Thread(incrementTask);
        
        t1.start();
        t2.start();
        
        try {
            t1.join();
            t2.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        System.out.println("Final counter value: " + counter.getValue());
    }

    private static class Counter {
        private int count = 0;
        
        public synchronized void increment() {
            count++;
        }
        
        public synchronized int getValue() {
            return count;
        }
    }

    public void testCompletableFuture() {
        System.out.println("Testing CompletableFuture operations...");
        
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return "Hello";
        });
        
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return "World";
        });
        
        CompletableFuture<String> combined = future1.thenCombine(future2, (s1, s2) -> s1 + " " + s2);
        
        try {
            String result = combined.get();
            System.out.println("Combined result: " + result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void testFileOperations() {
        try {
            Path tempFile = Files.createTempFile("test", ".txt");
            System.out.println("Created temp file: " + tempFile);
            
            // Write to file
            List<String> lines = Arrays.asList("Line 1", "Line 2", "Line 3");
            Files.write(tempFile, lines);
            
            // Read from file
            List<String> readLines = Files.readAllLines(tempFile);
            System.out.println("Read lines: " + readLines);
            
            // File properties
            long size = Files.size(tempFile);
            boolean exists = Files.exists(tempFile);
            System.out.println("File size: " + size + ", exists: " + exists);
            
            // Clean up
            Files.deleteIfExists(tempFile);
            
        } catch (Exception e) {
            System.err.println("File operation error: " + e.getMessage());
        }
    }

    public void testStreamProcessing() {
        List<String> words = Arrays.asList(
            "hello", "world", "java", "programming", "stream", "processing",
            "functional", "programming", "lambda", "expressions"
        );
        
        // Complex stream operations
        Map<Character, List<String>> groupedByFirstLetter = words.stream()
            .filter(word -> word.length() > 4)
            .map(String::toLowerCase)
            .distinct()
            .collect(Collectors.groupingBy(word -> word.charAt(0)));
        
        System.out.println("Grouped by first letter: " + groupedByFirstLetter);
        
        // Statistical operations
        IntSummaryStatistics stats = words.stream()
            .mapToInt(String::length)
            .summaryStatistics();
        
        System.out.println("Word length statistics: " + stats);
        
        // Parallel processing
        long parallelSum = words.parallelStream()
            .mapToInt(String::length)
            .sum();
        
        System.out.println("Parallel sum of lengths: " + parallelSum);
    }

    public void testExceptionHandling() {
        // Multiple exception types
        testDivision(10, 0);
        testArrayAccess(new int[5], 10);
        testNullPointer(null);
        testCustomException("invalid");
    }

    private void testDivision(int a, int b) {
        try {
            int result = a / b;
            System.out.println("Division result: " + result);
        } catch (ArithmeticException e) {
            System.err.println("Cannot divide by zero: " + e.getMessage());
        }
    }

    private void testArrayAccess(int[] arr, int index) {
        try {
            int value = arr[index];
            System.out.println("Array value: " + value);
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("Array index out of bounds: " + e.getMessage());
        }
    }

    private void testNullPointer(String str) {
        try {
            int length = str.length();
            System.out.println("String length: " + length);
        } catch (NullPointerException e) {
            System.err.println("Null pointer: " + e.getMessage());
        }
    }

    private void testCustomException(String input) {
        try {
            validateInput(input);
            System.out.println("Input is valid");
        } catch (CustomException e) {
            System.err.println("Custom exception: " + e.getMessage());
        }
    }

    private void validateInput(String input) throws CustomException {
        if ("invalid".equals(input)) {
            throw new CustomException("Input cannot be 'invalid'");
        }
    }

    private static class CustomException extends Exception {
        public CustomException(String message) {
            super(message);
        }
    }

    public void testResourceManagement() {
        // Try-with-resources
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Scanner resource managed automatically");
        } catch (Exception e) {
            System.err.println("Resource management error: " + e.getMessage());
        }
        
        // Manual resource management
        FileInputStream fis = null;
        try {
            // Simulated file operation
            System.out.println("Manual resource management example");
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        } finally {
            if (fis != null) {
                try {
                    fis.close();
                } catch (Exception e) {
                    System.err.println("Error closing resource: " + e.getMessage());
                }
            }
        }
    }

    // Utility methods for complex calculations
    public double calculateCompoundInterest(double principal, double rate, int time, int frequency) {
        return principal * Math.pow(1 + rate / frequency, frequency * time);
    }

    public boolean isPalindrome(String str) {
        String cleaned = str.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
        int left = 0, right = cleaned.length() - 1;
        while (left < right) {
            if (cleaned.charAt(left) != cleaned.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    public List<Integer> getPrimeFactors(int n) {
        List<Integer> factors = new ArrayList<>();
        for (int i = 2; i <= Math.sqrt(n); i++) {
            while (n % i == 0) {
                factors.add(i);
                n /= i;
            }
        }
        if (n > 1) factors.add(n);
        return factors;
    }

    public String encodeBase64(String input) {
        return Base64.getEncoder().encodeToString(input.getBytes());
    }

    public String decodeBase64(String encoded) {
        try {
            return new String(Base64.getDecoder().decode(encoded));
        } catch (IllegalArgumentException e) {
            return "Invalid Base64 string";
        }
    }

    public LocalDateTime getCurrentDateTime() {
        return LocalDateTime.now();
    }

    public String formatDateTime(LocalDateTime dateTime, String pattern) {
        return dateTime.format(java.time.format.DateTimeFormatter.ofPattern(pattern));
    }

    public long daysBetween(LocalDate start, LocalDate end) {
        return java.time.temporal.ChronoUnit.DAYS.between(start, end);
    }

    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public int getId() { return id; }
    public void setId(int id) { this.id = id; }
    public List<String> getItems() { return items; }
    public void setItems(List<String> items) { this.items = items; }
    public Map<String, Object> getProperties() { return properties; }
    public void setProperties(Map<String, Object> properties) { this.properties = properties; }

    @Override
    public String toString() {
        return "TestFile1{name='" + name + "', id=" + id + ", items=" + items + "}";
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        TestFile1 testFile1 = (TestFile1) obj;
        return id == testFile1.id && Objects.equals(name, testFile1.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, id);
    }
}
