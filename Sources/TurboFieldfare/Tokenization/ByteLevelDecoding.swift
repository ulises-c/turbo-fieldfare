import Foundation

/// The ByteLevel detokenization pipeline used by ChatML/Qwen tokenizers.
///
/// Qwen 3.6's `tokenizer.json` declares `decoder: ByteLevel` rather than
/// Gemma's `Sequence[Replace("▁" -> " "), ByteFallback, Fuse]`. In a ByteLevel
/// BPE every vocabulary token is already a string over GPT-2's byte alphabet:
/// each scalar stands for exactly one byte, so decoding is a scalar-to-byte
/// map followed by a UTF-8 assembly of the resulting byte stream.
///
/// Like `GemmaDecoding` this reproduces the declared decoder and stops there —
/// no `clean_up_tokenization_spaces` pass — so `decode(encode(x)) == x` and
/// batch and streaming decode agree by construction.
enum ByteLevelDecoding {
    /// GPT-2's `bytes_to_unicode` table, inverted: the scalar a byte is
    /// rendered as maps back to that byte. Printable ASCII/Latin-1 ranges keep
    /// their own code point; every other byte `n` is displaced to `256 + i`.
    static let scalarToByte: [Unicode.Scalar: UInt8] = {
        var direct: [UInt8] = []
        direct.append(contentsOf: UInt8(ascii: "!")...UInt8(ascii: "~"))
        direct.append(contentsOf: UInt8(0xA1)...UInt8(0xAC))
        direct.append(contentsOf: UInt8(0xAE)...UInt8(0xFF))

        var table: [Unicode.Scalar: UInt8] = [:]
        let directSet = Set(direct)
        for byte in direct {
            table[Unicode.Scalar(byte)] = byte
        }
        var displaced = 0
        for value in 0...255 {
            let byte = UInt8(value)
            guard !directSet.contains(byte) else { continue }
            // Safe: 256 + displaced stays well inside the BMP for 256 bytes.
            table[Unicode.Scalar(256 + displaced)!] = byte
            displaced += 1
        }
        return table
    }()

    /// The bytes a token contributes. A scalar outside the byte alphabet is not
    /// representable in this encoding; it is passed through as its own UTF-8
    /// bytes so foreign text degrades to itself instead of being dropped.
    static func bytes(_ token: String) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(token.unicodeScalars.count)
        for scalar in token.unicodeScalars {
            if let byte = scalarToByte[scalar] {
                out.append(byte)
            } else {
                out.append(contentsOf: Array(String(scalar).utf8))
            }
        }
        return out
    }
}

/// UTF-8 assembly for a ByteLevel token stream.
///
/// A multi-byte codepoint can be split across several tokens, so bytes are
/// buffered until they form whole scalars. Every complete codepoint is emitted
/// as soon as its last byte arrives — unlike `ByteFallbackRun`, which must hold
/// a whole run because ByteFallback commits atomically — so the stream stays
/// live and buffers at most three bytes. A byte that can never begin or
/// continue a valid sequence decodes to one U+FFFD, matching the reference
/// decoder's lossy UTF-8 handling.
struct ByteLevelRun {
    private var bytes: [UInt8] = []

    /// Text these bytes contribute, holding back only an incomplete tail.
    mutating func push(_ newBytes: [UInt8]) -> String {
        bytes.append(contentsOf: newBytes)
        return drain()
    }

    /// Close the stream: any held tail can no longer be completed.
    mutating func commit() -> String {
        var text = drain()
        if !bytes.isEmpty {
            text += String(repeating: "\u{FFFD}", count: bytes.count)
            bytes.removeAll(keepingCapacity: true)
        }
        return text
    }

    /// Emit every complete codepoint, keeping a still-completable tail.
    private mutating func drain() -> String {
        var text = ""
        var index = 0
        while index < bytes.count {
            guard let length = Self.sequenceLength(bytes[index]) else {
                text += "\u{FFFD}"
                index += 1
                continue
            }
            guard index + length <= bytes.count else { break }
            let scalarBytes = Array(bytes[index..<(index + length)])
            if let decoded = String(bytes: scalarBytes, encoding: .utf8) {
                text += decoded
                index += length
            } else {
                text += "\u{FFFD}"
                index += 1
            }
        }
        if index > 0 { bytes.removeFirst(index) }
        return text
    }

    /// Total length of the UTF-8 sequence this lead byte opens.
    private static func sequenceLength(_ lead: UInt8) -> Int? {
        switch lead {
        case 0x00...0x7F: return 1
        case 0xC2...0xDF: return 2
        case 0xE0...0xEF: return 3
        case 0xF0...0xF4: return 4
        default: return nil // stray continuation, overlong lead, or > U+10FFFF
        }
    }
}
