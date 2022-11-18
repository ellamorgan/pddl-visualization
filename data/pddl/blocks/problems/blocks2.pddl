;; 125 states
;; 4 blocks
(define (problem medium)
(:domain BLOCKS)
(:objects A B C D - block)
(:INIT (CLEAR B) (ONTABLE D) (ON B C) (ON C A) (ON A D) (HANDEMPTY))
(:goal (AND (ON D C) (ON C A) (ON A B)))
)