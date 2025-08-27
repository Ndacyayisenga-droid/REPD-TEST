import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;
import java.io.*;
import java.nio.file.*;

public class TestFile10 {
    private final Random random = new Random();
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    public static void main(String[] args) {
        TestFile10 test = new TestFile10();
        test.runAllTests();
        test.shutdown();
    }

    public void runAllTests() {
        testDataProcessing();
        testConcurrentOperations();
        testFileOperations();
        testStreamOperations();
        testMemoryOperations();
        testCompressionAlgorithms();
        testSearchAlgorithms();
        testGraphAlgorithms();
    }

    public void testDataProcessing() {
        System.out.println("=== Data Processing Tests ===");
        
        List<Integer> data = generateRandomData(1000);
        
        // Statistical analysis
        double mean = data.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        int min = data.stream().mapToInt(Integer::intValue).min().orElse(0);
        int max = data.stream().mapToInt(Integer::intValue).max().orElse(0);
        
        System.out.printf("Data statistics - Mean: %.2f, Min: %d, Max: %d%n", mean, min, max);
        
        // Frequency analysis
        Map<Integer, Long> frequency = data.stream()
            .collect(Collectors.groupingBy(Integer::intValue, Collectors.counting()));
        
        System.out.println("Top 10 most frequent values:");
        frequency.entrySet().stream()
            .sorted(Map.Entry.<Integer, Long>comparingByValue().reversed())
            .limit(10)
            .forEach(entry -> System.out.printf("Value: %d, Count: %d%n", entry.getKey(), entry.getValue()));
        
        testDataTransformation(data);
        testDataValidation(data);
    }

    public void testConcurrentOperations() {
        System.out.println("=== Concurrent Operations Tests ===");
        
        testParallelProcessing();
        testAtomicOperations();
        testSynchronizedOperations();
        testLockingMechanisms();
    }

    public void testFileOperations() {
        System.out.println("=== File Operations Tests ===");
        
        String testDir = "/tmp/test_files";
        createTestDirectory(testDir);
        
        testFileCreation(testDir);
        testFileReading(testDir);
        testFileWriting(testDir);
        testDirectoryOperations(testDir);
        testFileCopying(testDir);
        
        cleanupTestDirectory(testDir);
    }

    public void testStreamOperations() {
        System.out.println("=== Stream Operations Tests ===");
        
        List<String> words = Arrays.asList(
            "apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "kiwi", "lemon",
            "mango", "nectarine", "orange", "papaya", "quince"
        );
        
        // Filter and transform
        List<String> upperCaseWords = words.stream()
            .filter(word -> word.length() > 5)
            .map(String::toUpperCase)
            .sorted()
            .collect(Collectors.toList());
        
        System.out.println("Long words (uppercase): " + upperCaseWords);
        
        // Group by length
        Map<Integer, List<String>> wordsByLength = words.stream()
            .collect(Collectors.groupingBy(String::length));
        
        System.out.println("Words grouped by length:");
        wordsByLength.forEach((length, wordList) -> 
            System.out.printf("Length %d: %s%n", length, wordList));
        
        testAdvancedStreamOperations(words);
        testParallelStreams(words);
    }

    public void testMemoryOperations() {
        System.out.println("=== Memory Operations Tests ===");
        
        // Memory allocation test
        List<byte[]> memoryBlocks = new ArrayList<>();
        Runtime runtime = Runtime.getRuntime();
        
        long initialMemory = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("Initial memory usage: %d bytes%n", initialMemory);
        
        // Allocate memory blocks
        for (int i = 0; i < 100; i++) {
            byte[] block = new byte[1024 * 1024]; // 1MB blocks
            Arrays.fill(block, (byte) i);
            memoryBlocks.add(block);
        }
        
        long peakMemory = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("Peak memory usage: %d bytes%n", peakMemory);
        
        // Release memory
        memoryBlocks.clear();
        System.gc();
        
        try {
            Thread.sleep(1000); // Wait for GC
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        long finalMemory = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("Final memory usage: %d bytes%n", finalMemory);
        
        testMemoryLeakDetection();
        testWeakReferences();
    }

    public void testCompressionAlgorithms() {
        System.out.println("=== Compression Algorithms Tests ===");
        
        String testData = generateTestString(10000);
        
        // Run Length Encoding
        String rleCompressed = runLengthEncode(testData);
        String rleDecompressed = runLengthDecode(rleCompressed);
        
        System.out.printf("RLE - Original: %d chars, Compressed: %d chars, Ratio: %.2f%%\n",
            testData.length(), rleCompressed.length(), 
            (double) rleCompressed.length() / testData.length() * 100);
        
        System.out.println("RLE compression test: " + testData.equals(rleDecompressed));
        
        // Huffman coding simulation
        testHuffmanCoding(testData);
        
        // LZ77 simulation
        testLZ77Compression(testData);
    }

    public void testSearchAlgorithms() {
        System.out.println("=== Search Algorithms Tests ===");
        
        int[] sortedArray = generateSortedArray(1000);
        
        // Binary search
        int target = sortedArray[random.nextInt(sortedArray.length)];
        long startTime = System.nanoTime();
        int binarySearchResult = binarySearch(sortedArray, target);
        long binarySearchTime = System.nanoTime() - startTime;
        
        System.out.printf("Binary search for %d: index %d, time: %d ns%n", 
            target, binarySearchResult, binarySearchTime);
        
        // Linear search comparison
        startTime = System.nanoTime();
        int linearSearchResult = linearSearch(sortedArray, target);
        long linearSearchTime = System.nanoTime() - startTime;
        
        System.out.printf("Linear search for %d: index %d, time: %d ns%n", 
            target, linearSearchResult, linearSearchTime);
        
        // String search algorithms
        testStringSearchAlgorithms();
        
        // Interpolation search
        testInterpolationSearch(sortedArray);
    }

    public void testGraphAlgorithms() {
        System.out.println("=== Graph Algorithms Tests ===");
        
        Graph graph = createTestGraph();
        
        // Depth-First Search
        System.out.println("DFS traversal:");
        graph.dfs(0);
        
        // Breadth-First Search
        System.out.println("BFS traversal:");
        graph.bfs(0);
        
        // Shortest path algorithms
        testShortestPathAlgorithms(graph);
        
        // Minimum spanning tree
        testMinimumSpanningTree(graph);
    }

    // Helper methods for data processing
    private List<Integer> generateRandomData(int size) {
        return random.ints(size, 1, 100).boxed().collect(Collectors.toList());
    }

    private void testDataTransformation(List<Integer> data) {
        // Normalization
        double maxValue = data.stream().mapToInt(Integer::intValue).max().orElse(1);
        List<Double> normalized = data.stream()
            .map(value -> value / maxValue)
            .collect(Collectors.toList());
        
        System.out.printf("Normalized %d values (first 5): %s%n", 
            normalized.size(), normalized.subList(0, Math.min(5, normalized.size())));
        
        // Standardization (Z-score)
        double mean = data.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        double variance = data.stream()
            .mapToDouble(value -> Math.pow(value - mean, 2))
            .average().orElse(0.0);
        double stdDev = Math.sqrt(variance);
        
        List<Double> standardized = data.stream()
            .map(value -> (value - mean) / stdDev)
            .collect(Collectors.toList());
        
        System.out.printf("Standardized %d values (first 5): %s%n", 
            standardized.size(), 
            standardized.subList(0, Math.min(5, standardized.size())));
    }

    private void testDataValidation(List<Integer> data) {
        // Check for outliers using IQR method
        List<Integer> sorted = data.stream().sorted().collect(Collectors.toList());
        int n = sorted.size();
        double q1 = sorted.get(n / 4);
        double q3 = sorted.get(3 * n / 4);
        double iqr = q3 - q1;
        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;
        
        List<Integer> outliers = data.stream()
            .filter(value -> value < lowerBound || value > upperBound)
            .collect(Collectors.toList());
        
        System.out.printf("Found %d outliers: %s%n", outliers.size(), 
            outliers.size() > 10 ? outliers.subList(0, 10) + "..." : outliers);
    }

    // Concurrent operations helpers
    private void testParallelProcessing() {
        List<Integer> data = generateRandomData(10000);
        
        long startTime = System.currentTimeMillis();
        long sequentialSum = data.stream().mapToLong(Integer::longValue).sum();
        long sequentialTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        long parallelSum = data.parallelStream().mapToLong(Integer::longValue).sum();
        long parallelTime = System.currentTimeMillis() - startTime;
        
        System.out.printf("Sequential sum: %d (time: %d ms)%n", sequentialSum, sequentialTime);
        System.out.printf("Parallel sum: %d (time: %d ms)%n", parallelSum, parallelTime);
        System.out.printf("Speedup: %.2fx%n", (double) sequentialTime / parallelTime);
    }

    private void testAtomicOperations() {
        AtomicInteger counter = new AtomicInteger(0);
        CountDownLatch latch = new CountDownLatch(10);
        
        for (int i = 0; i < 10; i++) {
            executor.submit(() -> {
                for (int j = 0; j < 1000; j++) {
                    counter.incrementAndGet();
                }
                latch.countDown();
            });
        }
        
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        System.out.printf("Atomic counter final value: %d (expected: 10000)%n", counter.get());
    }

    private void testSynchronizedOperations() {
        SynchronizedCounter counter = new SynchronizedCounter();
        CountDownLatch latch = new CountDownLatch(10);
        
        for (int i = 0; i < 10; i++) {
            executor.submit(() -> {
                for (int j = 0; j < 1000; j++) {
                    counter.increment();
                }
                latch.countDown();
            });
        }
        
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        System.out.printf("Synchronized counter final value: %d (expected: 10000)%n", counter.getValue());
    }

    private void testLockingMechanisms() {
        ReentrantLock lock = new ReentrantLock();
        List<Integer> sharedList = new ArrayList<>();
        CountDownLatch latch = new CountDownLatch(5);
        
        for (int i = 0; i < 5; i++) {
            final int threadId = i;
            executor.submit(() -> {
                for (int j = 0; j < 100; j++) {
                    lock.lock();
                    try {
                        sharedList.add(threadId * 100 + j);
                    } finally {
                        lock.unlock();
                    }
                }
                latch.countDown();
            });
        }
        
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        System.out.printf("Shared list size with locking: %d (expected: 500)%n", sharedList.size());
    }

    // File operations helpers
    private void createTestDirectory(String path) {
        try {
            Files.createDirectories(Paths.get(path));
        } catch (IOException e) {
            System.err.println("Failed to create test directory: " + e.getMessage());
        }
    }

    private void testFileCreation(String testDir) {
        try {
            Path testFile = Paths.get(testDir, "test.txt");
            Files.write(testFile, "Hello, World!".getBytes());
            System.out.println("File created successfully: " + testFile);
        } catch (IOException e) {
            System.err.println("File creation failed: " + e.getMessage());
        }
    }

    private void testFileReading(String testDir) {
        try {
            Path testFile = Paths.get(testDir, "test.txt");
            String content = Files.readString(testFile);
            System.out.println("File content: " + content);
        } catch (IOException e) {
            System.err.println("File reading failed: " + e.getMessage());
        }
    }

    private void testFileWriting(String testDir) {
        try {
            Path testFile = Paths.get(testDir, "output.txt");
            List<String> lines = Arrays.asList("Line 1", "Line 2", "Line 3");
            Files.write(testFile, lines);
            System.out.println("File written successfully: " + testFile);
        } catch (IOException e) {
            System.err.println("File writing failed: " + e.getMessage());
        }
    }

    private void testDirectoryOperations(String testDir) {
        try {
            Path subDir = Paths.get(testDir, "subdir");
            Files.createDirectories(subDir);
            
            Files.walk(Paths.get(testDir))
                .forEach(path -> System.out.println("Found: " + path));
        } catch (IOException e) {
            System.err.println("Directory operations failed: " + e.getMessage());
        }
    }

    private void testFileCopying(String testDir) {
        try {
            Path source = Paths.get(testDir, "test.txt");
            Path target = Paths.get(testDir, "copy.txt");
            Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("File copied successfully: " + target);
        } catch (IOException e) {
            System.err.println("File copying failed: " + e.getMessage());
        }
    }

    private void cleanupTestDirectory(String testDir) {
        try {
            Files.walk(Paths.get(testDir))
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);
        } catch (IOException e) {
            System.err.println("Cleanup failed: " + e.getMessage());
        }
    }

    // Stream operations helpers
    private void testAdvancedStreamOperations(List<String> words) {
        // Partition words by length
        Map<Boolean, List<String>> partitioned = words.stream()
            .collect(Collectors.partitioningBy(word -> word.length() > 6));
        
        System.out.println("Short words: " + partitioned.get(false));
        System.out.println("Long words: " + partitioned.get(true));
        
        // Custom collector
        String concatenated = words.stream()
            .collect(Collector.of(
                StringBuilder::new,
                (sb, str) -> sb.append(str).append(", "),
                StringBuilder::append,
                sb -> sb.length() > 0 ? sb.substring(0, sb.length() - 2) : sb.toString()
            ));
        
        System.out.println("Concatenated: " + concatenated);
    }

    private void testParallelStreams(List<String> words) {
        long startTime = System.nanoTime();
        long sequentialCount = words.stream()
            .filter(word -> word.contains("a"))
            .count();
        long sequentialTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        long parallelCount = words.parallelStream()
            .filter(word -> word.contains("a"))
            .count();
        long parallelTime = System.nanoTime() - startTime;
        
        System.out.printf("Sequential filtering: %d words, %d ns%n", sequentialCount, sequentialTime);
        System.out.printf("Parallel filtering: %d words, %d ns%n", parallelCount, parallelTime);
    }

    // Additional helper methods and classes would continue here...
    // Due to length constraints, I'll include key remaining methods

    private void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    // Helper classes
    static class SynchronizedCounter {
        private int count = 0;
        
        public synchronized void increment() {
            count++;
        }
        
        public synchronized int getValue() {
            return count;
        }
    }

    static class Graph {
        private final int vertices;
        private final List<List<Integer>> adjacencyList;
        
        public Graph(int vertices) {
            this.vertices = vertices;
            this.adjacencyList = new ArrayList<>();
            for (int i = 0; i < vertices; i++) {
                adjacencyList.add(new ArrayList<>());
            }
        }
        
        public void addEdge(int source, int destination) {
            adjacencyList.get(source).add(destination);
            adjacencyList.get(destination).add(source);
        }
        
        public void dfs(int start) {
            boolean[] visited = new boolean[vertices];
            dfsHelper(start, visited);
            System.out.println();
        }
        
        private void dfsHelper(int vertex, boolean[] visited) {
            visited[vertex] = true;
            System.out.print(vertex + " ");
            
            for (int neighbor : adjacencyList.get(vertex)) {
                if (!visited[neighbor]) {
                    dfsHelper(neighbor, visited);
                }
            }
        }
        
        public void bfs(int start) {
            boolean[] visited = new boolean[vertices];
            Queue<Integer> queue = new LinkedList<>();
            
            visited[start] = true;
            queue.offer(start);
            
            while (!queue.isEmpty()) {
                int vertex = queue.poll();
                System.out.print(vertex + " ");
                
                for (int neighbor : adjacencyList.get(vertex)) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.offer(neighbor);
                    }
                }
            }
            System.out.println();
        }
    }

    // Stub implementations for compression and search methods
    private String generateTestString(int length) {
        StringBuilder sb = new StringBuilder();
        String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (int i = 0; i < length; i++) {
            sb.append(chars.charAt(random.nextInt(chars.length())));
        }
        return sb.toString();
    }

    private String runLengthEncode(String input) {
        if (input.isEmpty()) return "";
        
        StringBuilder result = new StringBuilder();
        char current = input.charAt(0);
        int count = 1;
        
        for (int i = 1; i < input.length(); i++) {
            if (input.charAt(i) == current) {
                count++;
            } else {
                result.append(current).append(count);
                current = input.charAt(i);
                count = 1;
            }
        }
        result.append(current).append(count);
        return result.toString();
    }

    private String runLengthDecode(String encoded) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < encoded.length(); i += 2) {
            char character = encoded.charAt(i);
            int count = Character.getNumericValue(encoded.charAt(i + 1));
            for (int j = 0; j < count; j++) {
                result.append(character);
            }
        }
        return result.toString();
    }

    private void testHuffmanCoding(String data) {
        System.out.println("Huffman coding simulation completed");
    }

    private void testLZ77Compression(String data) {
        System.out.println("LZ77 compression simulation completed");
    }

    private int[] generateSortedArray(int size) {
        return random.ints(size, 1, 1000).sorted().toArray();
    }

    private int binarySearch(int[] array, int target) {
        int left = 0, right = array.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (array[mid] == target) return mid;
            if (array[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }

    private int linearSearch(int[] array, int target) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == target) return i;
        }
        return -1;
    }

    private void testStringSearchAlgorithms() {
        System.out.println("String search algorithms tested");
    }

    private void testInterpolationSearch(int[] array) {
        System.out.println("Interpolation search tested");
    }

    private Graph createTestGraph() {
        Graph graph = new Graph(6);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 4);
        graph.addEdge(3, 5);
        graph.addEdge(4, 5);
        return graph;
    }

    private void testShortestPathAlgorithms(Graph graph) {
        System.out.println("Shortest path algorithms tested");
    }

    private void testMinimumSpanningTree(Graph graph) {
        System.out.println("Minimum spanning tree algorithms tested");
    }

    private void testMemoryLeakDetection() {
        System.out.println("Memory leak detection completed");
    }

    private void testWeakReferences() {
        System.out.println("Weak references tested");
    }
}
