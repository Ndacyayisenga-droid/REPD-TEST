package test;

import java.util.*;
import java.util.stream.*;
import java.io.*;
import java.nio.file.*;

public class TestFile3 {
    private List<String> dataList;
    private Map<String, Integer> scoreMap;
    private Set<String> uniqueItems;

    public TestFile3() {
        this.dataList = new ArrayList<>();
        this.scoreMap = new HashMap<>();
        this.uniqueItems = new HashSet<>();
    }

    public static void main(String[] args) {
        TestFile3 test = new TestFile3();
        test.runDataProcessingTests();
    }

    public void runDataProcessingTests() {
        testDataStructures();
        testFileProcessing();
        testStreamOperations();
        testArrayAlgorithms();
        testStringOperations();
    }

    public void testDataStructures() {
        System.out.println("Testing data structures...");
        
        // Stack operations
        Stack<Integer> stack = new Stack<>();
        for (int i = 1; i <= 5; i++) {
            stack.push(i);
        }
        while (!stack.isEmpty()) {
            System.out.println("Popped: " + stack.pop());
        }
        
        // Queue operations
        Queue<String> queue = new LinkedList<>();
        queue.offer("first");
        queue.offer("second");
        queue.offer("third");
        
        while (!queue.isEmpty()) {
            System.out.println("Dequeued: " + queue.poll());
        }
        
        // Priority Queue
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        pq.addAll(Arrays.asList(3, 1, 4, 1, 5, 9, 2, 6));
        
        while (!pq.isEmpty()) {
            System.out.println("Priority: " + pq.poll());
        }
        
        // LinkedHashMap preserving insertion order
        Map<String, Integer> orderedMap = new LinkedHashMap<>();
        orderedMap.put("third", 3);
        orderedMap.put("first", 1);
        orderedMap.put("second", 2);
        
        System.out.println("Ordered map: " + orderedMap);
        
        // TreeSet for sorted elements
        TreeSet<String> sortedSet = new TreeSet<>();
        sortedSet.addAll(Arrays.asList("zebra", "apple", "banana", "cherry"));
        System.out.println("Sorted set: " + sortedSet);
    }

    public void testFileProcessing() {
        System.out.println("Testing file processing...");
        
        try {
            // Create temporary file
            Path tempFile = Files.createTempFile("test", ".txt");
            
            // Write data to file
            List<String> lines = Arrays.asList(
                "Line 1: Hello World",
                "Line 2: Java Programming",
                "Line 3: File Processing",
                "Line 4: Stream Operations",
                "Line 5: Data Analysis"
            );
            
            Files.write(tempFile, lines);
            
            // Read and process file
            List<String> readLines = Files.readAllLines(tempFile);
            System.out.println("Read " + readLines.size() + " lines");
            
            // Process file content
            Map<String, Long> wordCount = readLines.stream()
                .flatMap(line -> Arrays.stream(line.split("\\s+")))
                .map(word -> word.replaceAll("[^a-zA-Z]", "").toLowerCase())
                .filter(word -> !word.isEmpty())
                .collect(Collectors.groupingBy(word -> word, Collectors.counting()));
            
            System.out.println("Word count: " + wordCount);
            
            // File statistics
            long fileSize = Files.size(tempFile);
            System.out.println("File size: " + fileSize + " bytes");
            
            // Clean up
            Files.deleteIfExists(tempFile);
            
        } catch (IOException e) {
            System.err.println("File processing error: " + e.getMessage());
        }
    }

    public void testStreamOperations() {
        System.out.println("Testing stream operations...");
        
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Filter even numbers and square them
        List<Integer> evenSquares = numbers.stream()
            .filter(n -> n % 2 == 0)
            .map(n -> n * n)
            .collect(Collectors.toList());
        
        System.out.println("Even squares: " + evenSquares);
        
        // Parallel processing
        double average = numbers.parallelStream()
            .mapToInt(Integer::intValue)
            .average()
            .orElse(0.0);
        
        System.out.println("Average: " + average);
        
        // Grouping and partitioning
        Map<Boolean, List<Integer>> partitioned = numbers.stream()
            .collect(Collectors.partitioningBy(n -> n % 2 == 0));
        
        System.out.println("Partitioned: " + partitioned);
        
        // Complex reduction
        Optional<Integer> max = numbers.stream()
            .reduce(Integer::max);
        
        System.out.println("Maximum: " + max.orElse(-1));
        
        // Custom collector
        String concatenated = numbers.stream()
            .map(String::valueOf)
            .collect(Collectors.joining(", ", "[", "]"));
        
        System.out.println("Concatenated: " + concatenated);
    }

    public void testArrayAlgorithms() {
        System.out.println("Testing array algorithms...");
        
        int[] array = {64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42};
        System.out.println("Original: " + Arrays.toString(array));
        
        // Selection sort
        int[] selectionSorted = array.clone();
        selectionSort(selectionSorted);
        System.out.println("Selection sorted: " + Arrays.toString(selectionSorted));
        
        // Insertion sort
        int[] insertionSorted = array.clone();
        insertionSort(insertionSorted);
        System.out.println("Insertion sorted: " + Arrays.toString(insertionSorted));
        
        // Binary search
        Arrays.sort(array);
        int target = 25;
        int index = Arrays.binarySearch(array, target);
        System.out.println("Binary search for " + target + ": " + index);
        
        // Array rotation
        int[] original = {1, 2, 3, 4, 5, 6, 7};
        int[] rotated = rotateArray(original, 3);
        System.out.println("Rotated by 3: " + Arrays.toString(rotated));
        
        // Find duplicates
        int[] withDuplicates = {1, 2, 3, 2, 4, 5, 3, 6};
        Set<Integer> duplicates = findDuplicates(withDuplicates);
        System.out.println("Duplicates: " + duplicates);
        
        // Two sum problem
        int[] nums = {2, 7, 11, 15};
        int targetSum = 9;
        int[] twoSumResult = twoSum(nums, targetSum);
        System.out.println("Two sum indices for " + targetSum + ": " + Arrays.toString(twoSumResult));
    }

    public void testStringOperations() {
        System.out.println("Testing string operations...");
        
        String text = "The quick brown fox jumps over the lazy dog";
        
        // Anagram detection
        String word1 = "listen";
        String word2 = "silent";
        boolean isAnagram = areAnagrams(word1, word2);
        System.out.println(word1 + " and " + word2 + " are anagrams: " + isAnagram);
        
        // Longest common prefix
        String[] strings = {"flower", "flow", "flight"};
        String commonPrefix = longestCommonPrefix(strings);
        System.out.println("Longest common prefix: " + commonPrefix);
        
        // String compression
        String compressed = compressString("aabcccccaaa");
        System.out.println("Compressed 'aabcccccaaa': " + compressed);
        
        // Remove duplicates
        String withoutDuplicates = removeDuplicateChars("programming");
        System.out.println("Without duplicates 'programming': " + withoutDuplicates);
        
        // Word frequency
        Map<String, Long> wordFreq = Arrays.stream(text.toLowerCase().split("\\s+"))
            .collect(Collectors.groupingBy(word -> word, Collectors.counting()));
        
        System.out.println("Word frequency: " + wordFreq);
        
        // Palindrome subsequence
        String palindromeText = "racecar";
        boolean isPalindrome = isPalindromeString(palindromeText);
        System.out.println(palindromeText + " is palindrome: " + isPalindrome);
    }

    // Algorithm implementations
    private void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            int temp = arr[minIdx];
            arr[minIdx] = arr[i];
            arr[i] = temp;
        }
    }

    private void insertionSort(int[] arr) {
        int n = arr.length;
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    private int[] rotateArray(int[] arr, int k) {
        int n = arr.length;
        k = k % n;
        int[] result = new int[n];
        
        for (int i = 0; i < n; i++) {
            result[(i + k) % n] = arr[i];
        }
        
        return result;
    }

    private Set<Integer> findDuplicates(int[] arr) {
        Set<Integer> seen = new HashSet<>();
        Set<Integer> duplicates = new HashSet<>();
        
        for (int num : arr) {
            if (!seen.add(num)) {
                duplicates.add(num);
            }
        }
        
        return duplicates;
    }

    private int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        
        return new int[0];
    }

    private boolean areAnagrams(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        
        char[] chars1 = s1.toLowerCase().toCharArray();
        char[] chars2 = s2.toLowerCase().toCharArray();
        
        Arrays.sort(chars1);
        Arrays.sort(chars2);
        
        return Arrays.equals(chars1, chars2);
    }

    private String longestCommonPrefix(String[] strings) {
        if (strings.length == 0) return "";
        
        String prefix = strings[0];
        for (int i = 1; i < strings.length; i++) {
            while (strings[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if (prefix.isEmpty()) return "";
            }
        }
        
        return prefix;
    }

    private String compressString(String str) {
        StringBuilder compressed = new StringBuilder();
        int count = 1;
        
        for (int i = 1; i < str.length(); i++) {
            if (str.charAt(i) == str.charAt(i - 1)) {
                count++;
            } else {
                compressed.append(str.charAt(i - 1)).append(count);
                count = 1;
            }
        }
        
        compressed.append(str.charAt(str.length() - 1)).append(count);
        
        return compressed.length() < str.length() ? compressed.toString() : str;
    }

    private String removeDuplicateChars(String str) {
        StringBuilder result = new StringBuilder();
        Set<Character> seen = new LinkedHashSet<>();
        
        for (char c : str.toCharArray()) {
            seen.add(c);
        }
        
        for (char c : seen) {
            result.append(c);
        }
        
        return result.toString();
    }

    private boolean isPalindromeString(String str) {
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

    // Matrix operations
    public void matrixOperations() {
        int[][] matrix1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        int[][] matrix2 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        
        int[][] sum = addMatrices(matrix1, matrix2);
        System.out.println("Matrix sum: " + Arrays.deepToString(sum));
        
        int[][] product = multiplyMatrices(matrix1, matrix2);
        System.out.println("Matrix product: " + Arrays.deepToString(product));
        
        int[][] transposed = transposeMatrix(matrix1);
        System.out.println("Transposed: " + Arrays.deepToString(transposed));
    }

    private int[][] addMatrices(int[][] a, int[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        int[][] result = new int[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        
        return result;
    }

    private int[][] multiplyMatrices(int[][] a, int[][] b) {
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

    // Tree operations
    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        
        TreeNode(int val) {
            this.val = val;
        }
    }

    public void testTreeOperations() {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        
        System.out.println("Inorder traversal:");
        inorderTraversal(root);
        System.out.println();
        
        System.out.println("Tree height: " + getTreeHeight(root));
        System.out.println("Tree size: " + getTreeSize(root));
    }

    private void inorderTraversal(TreeNode node) {
        if (node != null) {
            inorderTraversal(node.left);
            System.out.print(node.val + " ");
            inorderTraversal(node.right);
        }
    }

    private int getTreeHeight(TreeNode node) {
        if (node == null) return 0;
        return 1 + Math.max(getTreeHeight(node.left), getTreeHeight(node.right));
    }

    private int getTreeSize(TreeNode node) {
        if (node == null) return 0;
        return 1 + getTreeSize(node.left) + getTreeSize(node.right);
    }

    // Graph operations
    public void testGraphOperations() {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        graph.put(0, Arrays.asList(1, 2));
        graph.put(1, Arrays.asList(0, 3));
        graph.put(2, Arrays.asList(0, 4));
        graph.put(3, Arrays.asList(1));
        graph.put(4, Arrays.asList(2));
        
        System.out.println("DFS from node 0:");
        dfsTraversal(graph, 0, new HashSet<>());
        System.out.println();
        
        System.out.println("BFS from node 0:");
        bfsTraversal(graph, 0);
        System.out.println();
    }

    private void dfsTraversal(Map<Integer, List<Integer>> graph, int node, Set<Integer> visited) {
        visited.add(node);
        System.out.print(node + " ");
        
        for (int neighbor : graph.getOrDefault(node, Collections.emptyList())) {
            if (!visited.contains(neighbor)) {
                dfsTraversal(graph, neighbor, visited);
            }
        }
    }

    private void bfsTraversal(Map<Integer, List<Integer>> graph, int start) {
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
    }

    // Getters and setters
    public List<String> getDataList() { return dataList; }
    public void setDataList(List<String> dataList) { this.dataList = dataList; }
    public Map<String, Integer> getScoreMap() { return scoreMap; }
    public void setScoreMap(Map<String, Integer> scoreMap) { this.scoreMap = scoreMap; }
    public Set<String> getUniqueItems() { return uniqueItems; }
    public void setUniqueItems(Set<String> uniqueItems) { this.uniqueItems = uniqueItems; }

    @Override
    public String toString() {
        return "TestFile3{dataList=" + dataList + ", scoreMap=" + scoreMap + "}";
    }
}
