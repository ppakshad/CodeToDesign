# Code To Design Paper 
#Puya Pakshad
#Created at 11/4/2025
import os
import sys
import shutil
import subprocess
from pathlib import Path
# --- add near the other imports ---
import json
import getpass
import platform

# --- add near other configs ---
GPT5_MODEL = "gpt-5"               #  LLM MODEL CONFIG
PROMPT_FILENAME = "usecaseprompt.txt"
API_KEY_FILE = "openai_api_key.txt"  # optional local file next to project.py



# =========================
# Config (adjust if needed)
# =========================
JOERN_IMAGE = "ghcr.io/joernio/joern:nightly"
SC_SCRIPT_NAME = "usecase_table_metrics_v2.sc"  # target .sc file name
USE_DEFAULT_TRAVERSER = True  # set True to auto-write DEFAULT_SC_SCRIPT if .sc is missing

# Minimal, safe default traverser (writes a tiny JSON proving traversal works)
DEFAULT_SC_SCRIPT = r"""
/** usecase_table_metrics_v2.sc (fixed)
 * Static use-case oriented metrics over Joern CPG for C/C++.
 * Keeps v1 metrics + your research metrics (BRCG-ish, CFG stats, domain terms, TF-IDF, etc.).
 *
 * Env:
 *   CPG_PATH  (default: /work/out/cpg.bin)
 *   OUT_JSON  (default: /work/out/usecase_table_metrics_v2.json)
 *   UCD_TERMS (optional, comma-separated list of UCD terms to match)
 */

import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
import io.shiftleft.semanticcpg.language.locationCreator   // line/locations helpers
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.collection.mutable
import scala.util.Try

// ---------- Config ----------
val cpgPath = sys.env.getOrElse("CPG_PATH", "/work/out/cpg.bin")
val outPath = sys.env.getOrElse("OUT_JSON", "/work/out/usecase_table_metrics_v2.json")
val highCouplingThreshold = 10
val ucdTerms: Seq[String] = sys.env.get("UCD_TERMS").map(_.split(",").toSeq.map(_.trim).filter(_.nonEmpty)).getOrElse(Seq.empty)

// ---------- Load ----------
val cpg = CpgLoader.load(cpgPath)

// ---------- JSON helpers ----------
def esc(s: String): String =
  Option(s).getOrElse("").flatMap {
    case '"'  => "\\\""
    case '\\' => "\\\\"
    case '\b' => "\\b"
    case '\f' => "\\f"
    case '\n' => "\\n"
    case '\r' => "\\r"
    case '\t' => "\\t"
    case c if c < ' ' => "\\u%04x".format(c.toInt)
    case c => s"$c"
  }
def jstr(s:String) = "\"" + esc(s) + "\""
def jbool(b:Boolean) = if (b) "true" else "false"
def jint(i:Int) = i.toString
def jarr[T](xs:Iterable[T])(f:T=>String) = xs.map(f).mkString("[",",","]")
def jmap[V](m:Map[String,V])(f:V=>String) = "{"+m.toSeq.sortBy(_._1).map{case(k,v)=> jstr(k)+":"+f(v)}.mkString(",")+"}"
def baseName(p:String) = p.split("[/\\\\]").lastOption.getOrElse(p)

// ---------- Recognizers ----------
val entryPointNames = List("main","WinMain","DllMain")
val ioFile    = Set("fopen","freopen","fclose","fread","fwrite","fprintf","fscanf","open","close","read","write")
val ioConsole = Set("printf","puts","scanf","gets","getchar","putchar","cin","cout","cerr")
val ioNet     = Set("socket","connect","send","recv","bind","listen","accept","sendto","recvfrom")
val ioEnv     = Set("getenv","putenv","setenv","unsetenv")
val knownUnsafe = Set("gets","strcpy","strcat","sprintf","vsprintf","scanf")
val exitFuncs   = Set("exit","abort","_exit","quick_exit")
val nameClues   = List("login","logout","auth","register","create","delete","update","add","remove","deposit","withdraw","transfer","search","upload","download")
val inputNames  = Set("argv","argc","input","buf","line","user","password","username")

// ---------- Functions (robust) ----------
val allMethods = cpg.method.l
val definedMethods: List[Method] = allMethods.filter(m => m.ast.isBlock.l.nonEmpty || !m.isExternal)

val functionNames     = definedMethods.map(_.name).filter(_.nonEmpty).distinct.sorted
val functionFullNames = definedMethods.map(_.fullName).filter(_.nonEmpty).distinct.sorted
val functionSigs      = definedMethods.flatMap(m => Option(m.signature).map(_.toString)).filter(_.nonEmpty).distinct.sorted

// Build ["name(){...}", ...] by concatenating method name + first block code
def methodBody(m: Method): String =
  m.ast.isBlock.code.l.headOption.getOrElse("")

def inlineOneLine(s: String): String =
  Option(s).getOrElse("").replaceAll("\\s+", " ").trim

val functionListWithBodyJsonArray: String =
  definedMethods.map { m =>
    val name = Option(m.name).getOrElse("")
    val body = methodBody(m)                         // e.g., "{ ... }"
    val joined = inlineOneLine(name + body)          // "main(){...}"
    jstr(joined)
  }.mkString("[", ",", "]")


// ---------- Entry points ----------
val entryMethods   = cpg.method.nameExact(entryPointNames:_*).l
val publicMethods  = cpg.method.where(_.modifier.modifierType("PUBLIC")).l
val entryPointFullNames = (entryMethods.map(_.fullName) ++ publicMethods.map(_.fullName)).distinct.sorted

// ---------- Call graph (caller → callee) ----------
case class Edge(caller:String, callee:String, file:String, line:Option[Int])
val edges: List[Edge] = cpg.call.l.map { call =>
  val caller = call.method.fullName
  val callee = Option(call.methodFullName).getOrElse(call.name)
  val file   = call.file.name.headOption.getOrElse("")
  val line   = call.lineNumber.map(_.toInt)
  Edge(caller, callee, file, line)
}.distinct

val callersByCallee: Map[String,Set[String]] = edges.groupBy(_.callee).view.mapValues(_.map(_.caller).toSet).toMap
val calleesByCaller: Map[String,Set[String]] = edges.groupBy(_.caller).view.mapValues(_.map(_.callee).toSet).toMap

// ---------- BRCG-ish: Branch statements & call connectivity ----------
def isIfCode(s:String) = s.trim.toLowerCase.startsWith("if")
def isSwitchCode(s:String) = s.trim.toLowerCase.startsWith("switch")
def isLoopCode(s:String) = { val t=s.trim.toLowerCase; t.startsWith("for")||t.startsWith("while")||t.startsWith("do") }

case class BranchInfo(kind:String, code:String, file:String, line:Option[Int])
def branchInfos(m:Method): List[BranchInfo] = {
  val cs = m.controlStructure.l
  cs.map { n =>
    val c = Option(n.code).getOrElse("")
    val kind =
      if (isIfCode(c)) "if"
      else if (isSwitchCode(c)) "switch"
      else if (isLoopCode(c)) "loop"
      else if (c.trim.toLowerCase.startsWith("else")) "else"
      else "other"
    BranchInfo(kind, c, n.file.name.headOption.getOrElse(""), n.lineNumber.map(_.toInt))
  }
}

// Heuristic pruning of branches: keep "meaningful" ones
val branchKeywords = Set("error","fail","invalid","retry","menu","option","case","success","ok","auth","login","permission","granted","denied")
def meaningfulBranch(b:BranchInfo, m:Method): Boolean = {
  val low = b.code.toLowerCase
  val mentionsInput = inputNames.exists(low.contains)
  val mentionsKeyword = branchKeywords.exists(low.contains)
  val methodHasIO = m.call.name.l.exists(n => ioConsole(n.toLowerCase) || ioFile(n.toLowerCase) || ioNet(n.toLowerCase) || ioEnv(n.toLowerCase))
  mentionsInput || mentionsKeyword || methodHasIO
}

// ---------- CFG traversal approximations ----------
case class CfgStats(nodes:Int, edges:Int, backEdges:Int, longestAcyclic:Int, basicSegments:Int)

def cfgStats(m:Method): CfgStats = {
  val nodes: List[CfgNode] = m.cfgNode.l.collect { case n: CfgNode => n }.distinct
  val ids: Set[Long] = nodes.map(_.id).toSet

  val rawEdges: List[(Long, Long)] =
    nodes.flatMap(n => n._cfgOut.l.collect{ case nn: CfgNode => (n.id, nn.id) })  // <- was n.cfgNext.l ...

  // back-edge heuristic: successor.line < current.line
  def ln(id: Long): Option[Int] =
    nodes.find(_.id == id).flatMap(_.lineNumber).map(_.toInt)

  val backEdges: Int = rawEdges.count { case (a,b) =>
    (for { la <- ln(a); lb <- ln(b) } yield lb < la).getOrElse(false)
  }


  val dag: Map[Long, Set[Long]] =
    rawEdges
      .filterNot { case (a,b) => (for { la <- ln(a); lb <- ln(b) } yield lb < la).getOrElse(false) }
      .groupBy(_._1).view.mapValues(_.map(_._2).toSet).toMap


  val memo = scala.collection.mutable.HashMap.empty[Long, Int]
  def dfs(u: Long, seen: Set[Long]): Int = {
    if (seen.contains(u)) 0
    else memo.getOrElseUpdate(
      u, dag.getOrElse(u, Set.empty).map(v => 1 + dfs(v, seen + u)).foldLeft(0)(Math.max)
    )
  }

  val longest: Int = ids.map(id => dfs(id, Set.empty)).foldLeft(0)(Math.max)


  val outDeg: Map[Long, Int] = rawEdges.groupBy(_._1).view.mapValues(_.size).toMap
  val inDeg : Map[Long, Int] = rawEdges.groupBy(_._2).view.mapValues(_.size).toMap

  def isBranchNodeId(id: Long): Boolean = {
    val code = nodes.find(_.id == id).flatMap(n => Option(n.code)).getOrElse("").trim.toLowerCase
    code.startsWith("if") || code.startsWith("switch") || code.startsWith("case") || code.startsWith("default")
  }

  val basicSegments = ids.count { id =>
    val o = outDeg.getOrElse(id, 0); val i = inDeg.getOrElse(id, 0)
    (i != 1) || isBranchNodeId(id)
  }

  CfgStats(nodes.size, rawEdges.size, backEdges, longest, basicSegments)
}


// ---------- Control Flow (project totals) ----------
val branchCount = cpg.controlStructure.code.l.count(isIfCode)
val loopCount   = cpg.controlStructure.code.l.count(isLoopCode)
val switchCount = cpg.controlStructure.code.l.count(isSwitchCode)
val errorExitTotal = cpg.call.nameExact(exitFuncs.toSeq:_*).size

// ---------- Data / Artifacts ----------
val cliArgsUse = cpg.identifier.nameExact("argv","argc").size

def locals(m:Method) = m.local.name.l.toSet
def globalsUsed(m:Method) = {
  val locs = locals(m); val pars = m.parameter.name.l.toSet
  m.ast.isIdentifier.name.l.filterNot(n => locs.contains(n) || pars.contains(n) || n=="this").toSet
}
val sharedVarToFuncs: Map[String,Set[String]] =
  definedMethods.foldLeft(Map.empty[String, Set[String]]){ (acc,m) =>
    globalsUsed(m).foldLeft(acc){ case (a, g) => a.updated(g, a.getOrElse(g, Set.empty) + m.fullName) }
  }
val sharedArtifacts = sharedVarToFuncs.keySet.toSeq.sorted

def stringArgsOf(names:Set[String]) =
  cpg.call.nameExact(names.toSeq:_*).argument.isLiteral.code.l
    .map(_.stripPrefix("\"").stripSuffix("\""))
val filesTouched     = stringArgsOf(ioFile).distinct.sorted
val networkEndpoints = stringArgsOf(ioNet).distinct.sorted

// ---------- I/O counts ----------
val consoleInCount  = cpg.call.nameExact("scanf","gets","getchar","cin").size
val consoleOutCount = cpg.call.nameExact("printf","puts","putchar","cout","cerr").size
val fileIOCount     = cpg.call.name.l.count(ioFile.contains)
val netIOCount      = cpg.call.name.l.count(ioNet.contains)
val envIOCount      = cpg.call.name.l.count(ioEnv.contains)

// ---------- Name/Text Cues + Domain terms ----------
def splitTokens(s:String): Seq[String] = {
  val basic = s.replaceAll("[^A-Za-z0-9_]+"," ")
    .replaceAll("([a-z])([A-Z])","$1 $2")
    .toLowerCase.split("\\s+").toSeq.filter(_.nonEmpty)
  basic.flatMap(_.split("_")).filter(t => t.length>=3)
}
val stop = Set("int","char","void","long","short","double","float","const","static","class","struct","return","include","using","std","namespace","bool","true","false","null")

// comments inside method span — get comments by file via File → AST → Comment
def functionComments(m: Method): Seq[String] = {
  val f     = m.file.name.headOption.getOrElse("")
  val start = m.lineNumber.map(_.toInt).getOrElse(Int.MinValue)
  val end   = m.lineNumberEnd.map(_.toInt).getOrElse(Int.MaxValue)

  cpg.comment.l
    .filter { c =>
      // filename/lineNumber provided by `import io.shiftleft.semanticcpg.language.locationCreator`
      val sameFile =
        scala.util.Try(c.filename).toOption.contains(f) ||         // preferred
        scala.util.Try(c.location.filename).toOption.contains(f)   // fallback on some builds
      val inSpan   = c.lineNumber.exists(ln => ln >= start && ln <= end)
      sameFile && inSpan
    }
    .flatMap(c => Option(c.code))
}


// field identifier names (prefer canonicalName)
def memberAccessNames(m:Method): Seq[String] =
  m.ast.isFieldIdentifier.l.flatMap(fi => Option(fi.canonicalName))

def bagOfWords(m:Method): Seq[String] = {
  val idents = m.ast.isIdentifier.name.l
  val names  = Seq(m.name) ++ memberAccessNames(m)
  val cmts   = functionComments(m)
  (idents ++ names ++ cmts).flatMap(splitTokens).filterNot(stop)
}

// TF-IDF per function
val funcBags: Map[String, Seq[String]] = definedMethods.map(m => m.fullName -> bagOfWords(m)).toMap
val df: Map[String, Int] = funcBags.values.flatMap(_.distinct).toSeq.groupBy(identity).view.mapValues(_.size).toMap
val Ndocs = funcBags.size.max(1)
def tfidf(fn:String, k:String): Double = {
  val tf = funcBags.getOrElse(fn, Seq.empty).count(_==k).toDouble
  val idf = Math.log((Ndocs.toDouble + 1.0) / (df.getOrElse(k,0).toDouble + 1.0))
  tf * idf
}
def topTerms(fn:String, k:Int=10): Seq[(String,Double)] =
  funcBags.getOrElse(fn, Seq.empty).distinct
    .map(t => (t, tfidf(fn,t))).sortBy(-_._2).take(k)

// Optional similarity to UCD terms (substring match over tokens)
def ucdMatchesFor(fn:String): Map[String, Seq[String]] = {
  if (ucdTerms.isEmpty) Map.empty
  else {
    val toks = funcBags.getOrElse(fn, Seq.empty).distinct
    val m = ucdTerms.map { u =>
      val uLow = u.toLowerCase
      val hits = toks.filter(t => t.contains(uLow) || uLow.contains(t))
      u -> hits
    }.filter(_._2.nonEmpty).toMap
    m
  }
}

val nameCueMap = cpg.method.name.l.flatMap { n =>
  val low = n.toLowerCase; nameClues.filter(low.contains).map(_ -> n)
}.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap
val stringCueMap = cpg.literal.code.l.flatMap { s =>
  val low = s.toLowerCase; nameClues.filter(low.contains).map(_ -> s)
}.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap

// ---------- Complexity & Workflow ----------
def cyclomaticApprox(m:Method): Int = {
  val cs = m.controlStructure.code.l
  1 + cs.count(isIfCode) + cs.count(isLoopCode) + cs.count(isSwitchCode)
}
val perFuncComplexity = definedMethods.map { m =>
  val rec = m.call.methodFullName.l.contains(m.fullName)
  (m.fullName, cyclomaticApprox(m), rec)
}

// ---------- Cross-Module / API ----------
def fileOfMethod(fn:String) = cpg.method.fullNameExact(fn).file.name.headOption
val crossModuleEdges = edges.filter { e =>
  (for { cf <- fileOfMethod(e.caller); tf <- fileOfMethod(e.callee) } yield baseName(cf) != baseName(tf)).getOrElse(false)
}
val publicApiList = publicMethods.map(_.fullName).distinct.sorted

// ---------- Security ----------
val unsafeCallsList = cpg.call.name.l.filter(knownUnsafe.contains).distinct.sorted
val degrees = functionFullNames.map(fn => fn -> (calleesByCaller.getOrElse(fn,Set()).size + callersByCallee.getOrElse(fn,Set()).size)).toMap
val highCouplingFunctions = degrees.filter(_._2 >= highCouplingThreshold).keys.toSeq.sorted

def unvalidatedInputHeuristic(m:Method): Int = {
  val ifs = m.controlStructure.isIf.code.l
  val hasElse = m.controlStructure.code.l.exists(_.trim.toLowerCase.startsWith("else"))
  val suspicious = ifs.count { s =>
    val low = s.toLowerCase
    inputNames.exists(low.contains) || low.contains("scanf") || low.contains("gets") || low.contains("cin")
  }
  if (!hasElse) suspicious else 0
}
val unvalidatedSum = definedMethods.map(unvalidatedInputHeuristic).sum

// ---------- Per-function enrichments ----------
case class FuncEnrich(
  fullName:String,
  file:String,
  line:Option[Int],
  branches:Int, prunedBranches:Int,
  cfg: CfgStats,
  topTerms: Seq[(String,Double)],
  ucdMatches: Map[String,Seq[String]]
)
val funcEnrich: Seq[FuncEnrich] = definedMethods.map { m =>
  val brs = branchInfos(m).filter(b => b.kind=="if" || b.kind=="switch")
  val pruned = brs.filter(b => meaningfulBranch(b, m))
  FuncEnrich(
    fullName = m.fullName,
    file = m.file.name.headOption.getOrElse(""),
    line = m.lineNumber.map(_.toInt),
    branches = brs.size,
    prunedBranches = pruned.size,
    cfg = cfgStats(m),
    topTerms = topTerms(m.fullName, 10),
    ucdMatches = ucdMatchesFor(m.fullName)
  )
}




// ---------- Per-function I/O hints & neighbors (Trace Seeds) ----------
def methodByFullName(fn:String): Option[Method] =
  cpg.method.fullNameExact(fn).headOption

def stringArgsOfIn(m:Method, names:Set[String]): Seq[String] =
  m.call.nameExact(names.toSeq:_*).argument.isLiteral.code.l
    .map(_.stripPrefix("\"").stripSuffix("\"")).distinct

def ioHintsFor(m:Method): Seq[String] = {
  val fLits = stringArgsOfIn(m, ioFile)
  val nLits = stringArgsOfIn(m, ioNet)
  val eLits = stringArgsOfIn(m, ioEnv)
  val cin   = m.call.nameExact("scanf","gets","getchar","cin").size
  val cout  = m.call.nameExact("printf","puts","putchar","cout","cerr").size
  val acc = scala.collection.mutable.ArrayBuffer[String]()
  if (fLits.nonEmpty) acc += ("file:" + fLits.take(3).mkString(","))
  if (nLits.nonEmpty) acc += ("net:"  + nLits.take(3).mkString(","))
  if (eLits.nonEmpty) acc += ("env:"  + eLits.take(3).mkString(","))
  if (cin  > 0) acc += s"console_in:$cin"
  if (cout > 0) acc += s"console_out:$cout"
  acc.toSeq
}

def neighborsOf(fn:String, k:Int=6): Seq[String] = {
  val in  = callersByCallee.getOrElse(fn, Set.empty)
  val out = calleesByCaller.getOrElse(fn, Set.empty)
  (in ++ out).toSeq.distinct.sorted.take(k)
}

val traceSeedsJson =
  "{"+
    "\"PerFunction\":"+
      funcEnrich.map { f =>
        val mopt = methodByFullName(f.fullName)
        val ioHs = mopt.map(ioHintsFor).getOrElse(Seq.empty)
        val neigh = neighborsOf(f.fullName)
        val lex   = f.topTerms.map(_._1).take(8)
        "{"+
          "\"fullName\":"+jstr(f.fullName)+","+
          "\"file\":"+jstr(f.file)+","+
          "\"line\":"+f.line.map(_.toString).getOrElse("null")+","+
          "\"lexical_terms\":"+jarr(lex)(jstr)+","+
          "\"io_hints\":"+jarr(ioHs)(jstr)+","+
          "\"cfg_hints\":"+
            jarr(Seq(
              s"branches:${f.branches}",
              s"pruned:${f.prunedBranches}",
              s"cfg_nodes:${f.cfg.nodes}",
              s"cfg_edges:${f.cfg.edges}",
              s"back_edges:${f.cfg.backEdges}",
              s"longest_acyclic:${f.cfg.longestAcyclic}",
              s"basic_segments:${f.cfg.basicSegments}"
            ))(jstr)+","+
          "\"neighbors\":"+jarr(neigh)(jstr)+
        "}"
      }.mkString("[",",","]")+
  "}"





// ---------- Not available from static CPG ----------
val notAvailable = Map(
  "RuntimeValuesOfVariables" -> "requires dynamic tracing / execution",
  "InteractionEventsCRUD"    -> "requires developer activity traces (VCS/IDE logs)",
  "ActorArtefactMapping"     -> "requires developer perspective logs",
  "CRIRelevanceCentrality"   -> "requires developer navigation traces"
)

// ---------- Assemble JSON (stack-safe: build by sections) ----------

// Section builders (kept small to avoid staging recursion)
val entryPointsJson =
  "{"+
    "\"Count\":"+jint(entryPointFullNames.size)+","+
    "\"List\":"+jarr(entryPointFullNames)(jstr)+
  "}"

val functionsJson =
  "{"+
    "\"Count\":"+jint(functionNames.size)+","+
    "\"List\":"+functionListWithBodyJsonArray+","+
    "\"FullNames\":"+jarr(functionFullNames)(jstr)+","+
    "\"Signatures\":"+jarr(functionSigs)(jstr)+
  "}"

val callGraphJson =
  "{"+
    "\"Edges\":"+edges.map { e =>
      "{"+
        "\"caller\":"+jstr(e.caller)+","+
        "\"callee\":"+jstr(e.callee)+","+
        "\"file\":"+jstr(e.file)+","+
        "\"line\":"+(e.line.map(_.toString).getOrElse("null"))+
      "}"
    }.mkString("[",",","]")+
  "}"

val controlFlowJson =
  "{"+
    "\"Branch_Count\":"+jint(branchCount)+","+
    "\"Loop_Count\":"+jint(loopCount)+","+
    "\"Switch_Count\":"+jint(switchCount)+","+
    "\"ErrorExit.Paths\":"+jint(errorExitTotal)+
  "}"

val ioJson =
  "{"+
    "\"Console_Input_Uses\":"+jint(consoleInCount)+","+
    "\"Console_Output_Uses\":"+jint(consoleOutCount)+","+
    "\"File_IO_Uses\":"+jint(fileIOCount)+","+
    "\"Network_IO_Uses\":"+jint(netIOCount)+","+
    "\"EnvVar_Uses\":"+jint(envIOCount)+
  "}"

val dataArtifactsJson =
  "{"+
    "\"CLI.Args.Usage\":"+jint(cliArgsUse)+","+
    "\"SharedArtifacts\":"+jarr(sharedArtifacts)(jstr)+","+
    "\"Files.Touched\":"+jarr(filesTouched)(jstr)+","+
    "\"Network.Endpoints\":"+jarr(networkEndpoints)(jstr)+
  "}"

val nameTextCuesJson = {
  val nameCuesLocal = cpg.method.name.l.flatMap { n =>
    val low = n.toLowerCase; nameClues.filter(low.contains).map(_ -> n)
  }.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap

  val stringCuesLocal = cpg.literal.code.l.flatMap { s =>
    val low = s.toLowerCase; nameClues.filter(low.contains).map(_ -> s)
  }.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap

  "{"+
    "\"NameCues.Match\":"+jmap(nameCuesLocal)(vs => jarr(vs)(jstr))+","+
    "\"StringCues.Match\":"+jmap(stringCuesLocal)(vs => jarr(vs)(jstr))+
  "}"
}

val complexityWorkflowJson = {
  val perFunc =
    perFuncComplexity.map { case (fn,cpx,rec) =>
      "{"+
        "\"function\":"+jstr(fn)+","+
        "\"ComplexityApprox\":"+jint(cpx)+","+
        "\"Recursion.Flag\":"+jbool(rec)+
      "}"
    }.mkString("[",",","]")

  // longest path from entry (recomputed section-locally)
  val adj = calleesByCaller.view.mapValues(_.toSet).toMap
  val maxFromEntry = entryPointFullNames.map { n =>
    val memo = scala.collection.mutable.Map.empty[String, Int]
    def lf(x:String, seen:Set[String]): Int =
      memo.getOrElseUpdate(x, adj.getOrElse(x, Set.empty).filterNot(seen)
        .map(y => 1 + lf(y, seen + x)).foldLeft(0)(Math.max))
    n -> lf(n, Set.empty)
  }.toMap

  "{"+
    "\"ComplexityApprox.PerFunction\":"+perFunc+","+
    "\"Max_Path_Length_From_Entry\":"+jmap(maxFromEntry)(v => jint(v))+
  "}"
}

val crossModuleApiJson =
  "{"+
    "\"CrossModule.Calls\":"+jint(crossModuleEdges.size)+","+
    "\"PublicAPI.Count\":"+jint(publicApiList.size)+","+
    "\"PublicAPI.List\":"+jarr(publicApiList)(jstr)+
  "}"

val securityJson =
  "{"+
    "\"UnsafeCalls.List\":"+jarr(unsafeCallsList)(jstr)+","+
    "\"UnvalidatedInput.Heuristics\":"+jint(unvalidatedSum)+","+
    "\"HighCoupling.Functions\":"+jarr(highCouplingFunctions)(jstr)+
  "}"

val researchBrcgJson =
  "{"+
    "\"BranchStatements.PerFunction\":"+funcEnrich.map { f =>
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"branches\":"+jint(f.branches)+","+
        "\"prunedBranches\":"+jint(f.prunedBranches)+
      "}"
    }.mkString("[",",","]")+","+
    "\"ProcedureCalls.Edges\":"+edges.map { e =>
      "{"+
        "\"caller\":"+jstr(e.caller)+","+
        "\"callee\":"+jstr(e.callee)+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchCfgJson =
  "{"+
    "\"PerFunction\":"+funcEnrich.map { f =>
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"nodes\":"+jint(f.cfg.nodes)+","+
        "\"edges\":"+jint(f.cfg.edges)+","+
        "\"backEdgesHeuristic\":"+jint(f.cfg.backEdges)+","+
        "\"longestAcyclicPathApprox\":"+jint(f.cfg.longestAcyclic)+","+
        "\"basicSegments\":"+jint(f.cfg.basicSegments)+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchDomainTermsJson =
  "{"+
    "\"TopTFIDF.PerFunction\":"+funcEnrich.map { f =>
      val tops = f.topTerms.map { case (t,score) =>
        "{"+
          "\"term\":"+jstr(t)+","+
          "\"score\":"+jstr("%.5f".format(score))+
        "}"
      }.mkString("[",",","]")
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"TopTerms\":"+tops+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchCommentsJson =
  "{"+
    "\"HasComments.PerFunction\":"+definedMethods.map { m =>
      "{"+
        "\"function\":"+jstr(m.fullName)+","+
        "\"count\":"+jint(functionComments(m).size)+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchRelationsJson = {
  val inh = cpg.typeDecl.l.map { td =>
    "{"+
      "\"type\":"+jstr(Option(td.fullName).getOrElse(""))+","+
      "\"inherits\":"+jarr(td.inheritsFromTypeFullName.l.distinct.sorted)(jstr)+
    "}"
  }.mkString("[",",","]")

  val varToFns = jmap(sharedVarToFuncs.map { case (k,v) => k -> v.toSeq.sorted })(vs => jarr(vs)(jstr))

  "{"+
    "\"Inheritance\":"+inh+","+
    "\"SharedGlobals\":{"+
      "\"Variables\":"+jarr(sharedVarToFuncs.keys.toSeq.sorted)(jstr)+","+
      "\"VarToFunctions\":"+varToFns+
    "}"+
  "}"
}

val researchSimilarityJson =
  "{"+
    "\"Enabled\":"+jbool(ucdTerms.nonEmpty)+","+
    "\"UCDElements\":"+jarr(ucdTerms)(jstr)+","+
    "\"Matches.PerFunction\":"+funcEnrich.map { f =>
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"matches\":"+jmap(f.ucdMatches)(vs => jarr(vs)(jstr))+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchNotAvailJson = jmap(notAvailable)(jstr)

val researchJson =
  "{"+
    "\"BRCG\":"+researchBrcgJson+","+
    "\"CFGTraversal\":"+researchCfgJson+","+
    "\"DomainTerms\":"+researchDomainTermsJson+","+
    "\"Comments\":"+researchCommentsJson+","+
    "\"Relations\":"+researchRelationsJson+","+
    "\"SimilarityToUCD\":"+researchSimilarityJson+","+
    "\"NotAvailableFromStaticCPG\":"+researchNotAvailJson+
  "}"

val top =
  "{"+
    "\"EntryPoints\":"+entryPointsJson+","+
    "\"Functions\":"+functionsJson+","+
    "\"CallGraph\":"+callGraphJson+","+
    "\"ControlFlow\":"+controlFlowJson+","+
    "\"IO\":"+ioJson+","+
    "\"DataArtifacts\":"+dataArtifactsJson+","+
    "\"NameTextCues\":"+nameTextCuesJson+","+
    "\"ComplexityWorkflow\":"+complexityWorkflowJson+","+
    "\"CrossModuleAPI\":"+crossModuleApiJson+","+
    "\"Security\":"+securityJson+","+
    "\"TraceSeeds\":"+traceSeedsJson+","+
    "\"ResearchMetrics\":"+researchJson+
  "}"


Files.write(Paths.get(outPath), top.getBytes(StandardCharsets.UTF_8))
println("Wrote JSON to: "+outPath)
"""


# =========================
# Utilities
# =========================
def try_run(cmd, timeout=None):
    """Run subprocess w/ UTF-8 decoding to avoid cp1252 issues on Windows."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr=str(e))
    except Exception as e:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=str(e))

def docker_in_path() -> bool:
    return shutil.which("docker") is not None or shutil.which("docker.exe") is not None

# =========================
# Docker / Joern checks
# =========================
def test_connection_to_docker(verbose=True) -> bool:
    """Ensure docker CLI exists and engine is running."""
    if not docker_in_path():
        if verbose:
            print("Docker CLI not found in PATH.")
        return False

    ver = try_run(["docker", "--version"], timeout=10)
    if ver.returncode != 0:
        if verbose:
            print("'docker --version' failed:")
            print((ver.stderr or "").strip())
        return False
    if verbose:
        print("Docker CLI:", (ver.stdout or "").strip())

    info = try_run(["docker", "info"], timeout=20)
    if info.returncode != 0:
        if verbose:
            print("'docker info' failed: Docker engine not running?")
            err = (info.stderr or info.stdout or "").strip()
            if err:
                print(err)
        return False

    if verbose:
        print("Docker engine is running.")
    return True

def ensure_joern_image(image: str = JOERN_IMAGE, verbose=True) -> bool:
    probe = try_run(["docker", "image", "inspect", image], timeout=20)
    if probe.returncode == 0:
        if verbose:
            print(f"Image present: {image}")
        return True
    if verbose:
        print(f"⬇ Pulling image: {image}")
    pull = try_run(["docker", "pull", image], timeout=900)
    if pull.returncode != 0:
        if verbose:
            print("'docker pull' failed:")
            print((pull.stderr or pull.stdout or "").strip())
        return False
    if verbose:
        print("Image pulled successfully.")
    return True

def test_connection_to_joern(image: str = JOERN_IMAGE, verbose=True) -> bool:
    """Verify joern is invocable in the container (via --help)."""
    if not ensure_joern_image(image, verbose=verbose):
        return False

    attempts = [
        ["docker", "run", "--rm", "--entrypoint", "joern", image, "--help"],
        ["docker", "run", "--rm", image, "joern", "--help"],
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-lc", "joern --help || /opt/joern/joern --help"],
    ]

    last_err = ""
    for cmd in attempts:
        res = try_run(cmd, timeout=90)
        if res.returncode == 0:
            if verbose:
                print("Joern --help via:", " ".join(cmd))
                preview = "\n".join(((res.stdout or "") + "\n" + (res.stderr or "")).splitlines()[:20]).strip()
                if preview:
                    print(preview)
            return True
        last_err = (res.stderr or res.stdout or "").strip()

    if verbose:
        print("Could not invoke joern in container.")
        if last_err:
            print(last_err)
    return False

# =========================
# Paths / Export
# =========================
def compute_out_dir_from_project(project_code_dir: Path) -> Path:
    """Given '.../code' returns sibling '.../graphs'. If not named 'code', still use sibling 'graphs'."""
    project_code_dir = project_code_dir.resolve()
    if project_code_dir.name.lower() == "code":
        return project_code_dir.parent / "graphs"
    return project_code_dir.parent / "graphs"

def validate_project_dir(project_code_dir: Path) -> bool:
    if not project_code_dir.exists():
        print(f"Project path does not exist: {project_code_dir}")
        return False
    if not project_code_dir.is_dir():
        print(f"Project path is not a directory: {project_code_dir}")
        return False
    return True

def run_joern_export(image: str, project_code_dir: Path, out_dir: Path) -> bool:
    """
    Run joern-parse + joern-export (graphson) inside the container.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{str(project_code_dir)}:/work/project",
        "-v", f"{str(out_dir)}:/work/out",
        image,
        "bash", "-lc",
        "set -e; cd /work/out; "
        "joern-parse /work/project; "
        "rm -rf /work/out/cpg_graphson; "
        "joern-export --repr all --format graphson --out /work/out/cpg_graphson"
    ]

    print("\nRunning Joern parse + export ...")
    res = try_run(cmd, timeout=3600)

    if res.returncode == 0:
        print("Export completed.")
        print(f"Output: {out_dir / 'cpg_graphson'}")
        if (res.stdout or "").strip():
            print("\n=== STDOUT ===")
            print(res.stdout.strip())
        if (res.stderr or "").strip():
            print("\n=== STDERR ===")
            print(res.stderr.strip())
        return True

    print("Export failed.")
    if (res.stderr or "").strip():
        print("\n— Error —")
        print(res.stderr.strip())
    elif (res.stdout or "").strip():
        print("\n— Output —")
        print(res.stdout.strip())
    return False

# =========================
# Traversal (only write the .sc)
# =========================
def traversingGraph(
    out_dir: Path,
    script_text: str | None = None,
    script_filename: str = SC_SCRIPT_NAME,
    overwrite: bool = True,
    verbose: bool = True
) -> bool:
    """
    1) Ensure OUT dir exists
    2) Write the Scala script (UTF-8, no BOM) to OUT/script_filename
    3) No Docker run here (manual execution later)
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / script_filename
    if script_text is not None:
        if script_path.exists() and not overwrite:
            if verbose:
                print(f"ℹ Script already exists, skip writing: {script_path}")
        else:
            script_path.write_text(script_text, encoding="utf-8", newline="\n")
            if verbose:
                print(f"Wrote script: {script_path}")
    else:
        if not script_path.exists():
            if USE_DEFAULT_TRAVERSER:
                script_path.write_text(DEFAULT_SC_SCRIPT, encoding="utf-8", newline="\n")
                if verbose:
                    print(f"Wrote default script: {script_path}")
            else:
                print(f"Script not found and no script_text provided: {script_path}")
                return False
        else:
            if verbose:
                print(f"Script already exists: {script_path}")

    if verbose:
        print(f"Ready: .sc file is available at {script_path}")
    return True

# =========================
# Step 7: Execute Joern script (SC → JSON)
# =========================
def run_joern_script(
    image: str,
    out_dir: Path,
    script_filename: str = SC_SCRIPT_NAME,
    out_json_name: str = "usecase_table_metrics_v2.json",
    ucd_terms: str | None = None,
    verbose: bool = True,
    timeout: int = 3600,
) -> bool:
    """
    Run `joern --script` inside the container to produce the OUT_JSON file.
    - image: Docker image (e.g., ghcr.io/joernio/joern:nightly)
    - out_dir: the local `graphs` directory (mounted to /work/out)
    - script_filename: .sc file that must exist under out_dir
    - out_json_name: JSON output name (created under out_dir)
    - ucd_terms: optional comma-separated UCD terms to pass via env
    """
    out_dir = Path(out_dir).resolve()
    script_path = out_dir / script_filename
    cpg_bin = out_dir / "cpg.bin"
    out_json_path = out_dir / out_json_name

    # Basic checks
    if not script_path.exists():
        print(f"SC script not found: {script_path}")
        return False
    if not cpg_bin.exists():
        print(f"cpg.bin not found at {cpg_bin}. Run export step first.")
        return False

    # Build env string for bash -lc
    env_parts = [
        "CPG_PATH=/work/out/cpg.bin",
        f"OUT_JSON=/work/out/{out_json_name}",
    ]
    if ucd_terms:
        env_parts.append(f'UCD_TERMS="{ucd_terms}"')
    env_str = " ".join(env_parts)

    # NOTE: don't use -it in non-interactive runs
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{str(out_dir)}:/work/out",
        image,
        "bash", "-lc",
        f'{env_str} joern --script /work/out/{script_path.name}'
    ]

    if verbose:
        print("\n Running Joern script (SC → JSON) ...")
        print("Command:", " ".join(cmd))

    res = try_run(cmd, timeout=timeout)

    if res.returncode == 0:
        if verbose:
            if (res.stdout or "").strip():
                print("\n=== STDOUT ===")
                print(res.stdout.strip())
            if (res.stderr or "").strip():
                print("\n=== STDERR ===")
                print(res.stderr.strip())
            print(f"\n✅ JSON created at: {out_json_path}")
        return True

    print("Joern script execution failed.")
    if (res.stderr or "").strip():
        print("\n— Error —")
        print(res.stderr.strip())
    elif (res.stdout or "").strip():
        print("\n— Output —")
        print(res.stdout.strip())
    return False


def _looks_like_openai_key(s: str) -> bool:
    """Simple heuristic: 'sk-' prefix and reasonable length."""
    return isinstance(s, str) and s.strip().startswith("sk-") and len(s.strip()) > 20

def _persist_api_key_windows(key: str) -> bool:
    """Persist OPENAI_API_KEY for the current user on Windows. New shells will pick it up."""
    try:
        # Use PowerShell API so we don't depend on setx presence.
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Environment]::SetEnvironmentVariable('OPENAI_API_KEY','{}','User')".format(key)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return res.returncode == 0
    except Exception:
        return False

def ensure_openai_api_key(persist: bool = False,
                          key_path: Path | None = None,
                          interactive: bool = True) -> bool:
    """
    Simplified version: always set OPENAI_API_KEY from hardcoded value.
    Returns True if key was set.
    """
    import os

    hardcoded_key = ""  #API KEY

    if not hardcoded_key or not hardcoded_key.startswith("sk-"):
        print("Hardcoded API key is missing or invalid.")
        return False

    os.environ["OPENAI_API_KEY"] = hardcoded_key
    print("OPENAI_API_KEY set from hardcoded value.")
    return True


# =========================
# Step 8: Send CPG JSON + prompt to GPT-5 (Responses API)
# =========================
def send_metrics_to_llm(out_dir: Path,
                        prompt_path: Path | None = None,
                        model: str = os.environ.get("OPENAI_MODEL", "gpt-5"),
                        prefer_upload: bool = True,  
                        verbose: bool = True) -> dict | None:

    import json, re

    out_dir = Path(out_dir).resolve()
    cpg_json = out_dir / "usecase_table_metrics_v2.json"
    if not cpg_json.exists():
        print(f"JSON not found: {cpg_json}")
        return None

    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent / "usecaseprompt.txt"
    else:
        prompt_path = Path(prompt_path).resolve()
    if not prompt_path.exists():
        print(f"Prompt file not found: {prompt_path}")
        return None

    prompt_text = prompt_path.read_text(encoding="utf-8")
    json_text   = cpg_json.read_text(encoding="utf-8")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing. Set it before calling LLM.")
        return None

    instruction = (
        "You are a code analysis assistant. Read the provided CPG metrics JSON and the prompt text. "
        "Return EXACTLY one valid JSON object as the final answer, with no extra commentary."
    )

    def chunk_text(s: str, size: int = 80_000):
        return [s[i:i+size] for i in range(0, len(s), size)]

    json_chunks = chunk_text(json_text, 80_000)

    contents = [
        {"type": "input_text", "text": instruction},
        {"type": "input_text", "text": "PROMPT:\n" + prompt_text.strip()},
        {"type": "input_text", "text": f"CPG_JSON has {len(json_chunks)} chunk(s). Reconstruct in order."},
    ]
    for i, ch in enumerate(json_chunks, 1):
        contents.append({"type": "input_text", "text": f"CPG_JSON_PART {i}/{len(json_chunks)}:\n{ch}"})

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": contents}],
        )

        raw = getattr(resp, "output_text", None)
        if not raw:
            try:
                out = getattr(resp, "output", None)
                if out:
                    for blk in out:
                        for item in getattr(blk, "content", []):
                            if getattr(item, "type", "") in ("output_text", "text") and hasattr(item, "text"):
                                raw = item.text
                                break
            except Exception:
                pass
        if not raw:
            raw = str(resp)

        try:
            obj = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            obj = json.loads(m.group(0)) if m else {"raw": raw}

        if verbose:
            print("\n=== LLM JSON ===")
            #print(json.dumps(obj, ensure_ascii=False, indent=2)[:3000])

        return obj

    except Exception as e:
        print("LLM call failed.")
        print(str(e))
        return None


# ====== Global outputs from LLM ======
PLANTUML_TEXT = None   # type: str | None
USECASE_BINDINGS = None  # type: list | None

def extract_from_llm(llm_obj):
    
    global PLANTUML_TEXT, USECASE_BINDINGS
    PLANTUML_TEXT = None
    USECASE_BINDINGS = None

    if not isinstance(llm_obj, dict):
        return

    PLANTUML_TEXT = (
        llm_obj.get("plantuml")
        or llm_obj.get("PlantUML")
        or llm_obj.get("uml")
    )

    bindings = (
        llm_obj.get("UseCaseCodeBindings")
        or llm_obj.get("usecase_code_bindings")
        or llm_obj.get("bindings")
    )
    if isinstance(bindings, list):
        USECASE_BINDINGS = bindings



def func9_generate_plantuml_png(puml_text, out_dir):
    """
    Generate PlantUML diagram as PNG using the python-plantuml library.
    Requires: pip install plantuml
    It saves .puml first, then asks the PlantUML server to render a PNG next to it.
    """
    from pathlib import Path
    from plantuml import PlantUML

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    puml_path = out_dir / "usecase_usecases.puml"
    png_path  = out_dir / "usecase_usecases.png"

    puml_path.write_text(puml_text, encoding="utf-8")

    try:
        server = PlantUML(url="http://www.plantuml.com/plantuml/png/")
        result = server.processes([str(puml_path)])

        if png_path.exists():
            print(f"PlantUML PNG generated at: {png_path}")
            return png_path
        else:
            print("PlantUML server did not produce PNG. Raw result:", result)
            print(f".puml is saved at: {puml_path}")
            return None
    except Exception as e:
        print("Failed to render PlantUML via plantuml library")
        print(str(e))
        print(f"Saved .puml instead at: {puml_path}")
        return None


def func10_write_bindings_txt(bindings, out_file=None):

    import json
    from pathlib import Path

    if out_file is None:
        out_file = Path.cwd() / "testout.txt"
    else:
        out_file = Path(out_file)

    lines = []
    for idx, b in enumerate(bindings, 1):
        uc_id = str(b.get("uc_id", ""))
        label = str(b.get("label", ""))
        actors = ", ".join(b.get("actors", []) or [])
        trigger = str(b.get("trigger", ""))
        description = str(b.get("description", ""))

        pre = "\n    - ".join(b.get("preconditions", []) or [])
        post = "\n    - ".join(b.get("postconditions", []) or [])

        nf_raw = b.get("normal_flow", []) or []
        nf_items = []
        if isinstance(nf_raw, list):
            for s in nf_raw:
                if isinstance(s, str):
                    nf_items.append(s)
                elif isinstance(s, dict):
                    sid = s.get("id")
                    steps = s.get("steps")
                    if sid and steps:
                        for st in steps:
                            nf_items.append(f"{sid} {st}")
                    else:
                        nf_items.append(json.dumps(s, ensure_ascii=False))
                else:
                    nf_items.append(str(s))
        else:
            nf_items.append(str(nf_raw))
        nf = "\n    - ".join(nf_items)

        af_raw = b.get("alt_flows", []) or []
        af_items = []
        for af in af_raw:
            if isinstance(af, dict):
                aid = af.get("id", "")
                steps = af.get("steps", []) or []
                if steps:
                    for st in steps:
                        af_items.append(f"{aid} {st}")
            else:
                af_items.append(str(af))
        af = "\n    - ".join(af_items)

        ex_raw = b.get("exceptions", []) or []
        ex_items = []
        for ex in ex_raw:
            if isinstance(ex, dict):
                eid = ex.get("id", "")
                cause = ex.get("cause", "")
                resp = ex.get("system_response", "")
                state = ex.get("state_change", "")
                steps = ex.get("steps", []) or []
                head = f"{eid} cause={cause} response={resp} state={state}".strip()
                ex_items.append(head)
                for st in steps:
                    ex_items.append(f"    * {st}")
            else:
                ex_items.append(str(ex))
        ex = "\n    - ".join(ex_items)

        th = "\n    - ".join(b.get("threats", []) or [])

        bind = b.get("bindings", []) or []
        if bind:
            bind_lines = []
            for x in bind:
                bind_lines.append(
                    json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
                )
            bind_str = "\n    - " + "\n    - ".join(bind_lines)
        else:
            bind_str = ""

        block = f"""================ Use Case {idx} ================
UC_ID       : {uc_id}
Label       : {label}
Actors      : {actors}
Trigger     : {trigger}
Description : {description}

Preconditions:
    - {pre if pre else '(none)'}
Postconditions:
    - {post if post else '(none)'}

Normal Flow:
    - {nf if nf else '(none)'}

Alternative Flows:
    - {af if af else '(none)'}

Exceptions:
    - {ex if ex else '(none)'}

Threats:
    - {th if th else '(none)'}

Bindings:{bind_str if bind_str else ' (none)'}
"""
        lines.append(block)

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Wrote bindings to: {out_file}")
    return out_file







def func11_generate_security_report(
    prompt_path: Path | None = None,
    input_txt_path: Path | None = None,
    model: str = os.environ.get("OPENAI_MODEL", "gpt-5"),
    verbose: bool = True,
    chunk_size: int = 80_000,
) -> Path | None:
 
    import json, re
    from openai import OpenAI

    base_dir = Path(__file__).resolve().parent   
    out_json_path = base_dir / "securityReport.json"

    if prompt_path is None:
        prompt_path = base_dir / "securityprompt.txt"
    else:
        prompt_path = Path(prompt_path).resolve()

    if input_txt_path is None:
        input_txt_path = base_dir / "testout.txt"
    else:
        input_txt_path = Path(input_txt_path).resolve()

    if not prompt_path.exists():
        print(f"securityprompt.txt not found: {prompt_path}")
        return None
    if not input_txt_path.exists():
        print(f"input text not found: {input_txt_path}")
        return None

    prompt_text = prompt_path.read_text(encoding="utf-8")
    input_text  = input_txt_path.read_text(encoding="utf-8")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing.")
        return None

    instruction = (
        "You are a security analysis assistant. Read the provided PROMPT and INPUT_TEXT, "
        "and produce EXACTLY ONE valid JSON object as the final answer (no commentary)."
    )

    def chunk_text(s: str, size: int):
        return [s[i:i+size] for i in range(0, len(s), size)]

    input_chunks = chunk_text(input_text, chunk_size)

    contents = [
        {"type": "input_text", "text": instruction},
        {"type": "input_text", "text": "PROMPT:\n" + prompt_text.strip()},
        {"type": "input_text", "text": f"INPUT_TEXT has {len(input_chunks)} chunk(s). Reconstruct in order."},
    ]
    for i, ch in enumerate(input_chunks, 1):
        contents.append({"type": "input_text", "text": f"INPUT_TEXT_PART {i}/{len(input_chunks)}:\n{ch}"})

    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": contents}],
        )
        raw = getattr(resp, "output_text", None) or str(resp)

        try:
            obj = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            obj = json.loads(m.group(0)) if m else {"raw": raw}

        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"Security report saved to: {out_json_path}")
        return out_json_path

    except Exception as e:
        print("Security report generation failed.")
        print(str(e))
        return None




















# =========================
# Main
# =========================
def main():
    # Help Docker with Windows paths
    os.environ.setdefault("COMPOSE_CONVERT_WINDOWS_PATHS", "1")

    # 1) Docker health
    print("== Checking Docker connectivity ==")
    if not test_connection_to_docker(verbose=True):
        sys.exit(1)

    # 2) Joern availability
    print("\n== Checking Joern in Docker ==")
    if not test_connection_to_joern(image=JOERN_IMAGE, verbose=True):
        sys.exit(2)

    # 3) Ask for PROJECT (the `...\code` directory)
    print("\nEnter your C/C++ project 'code' folder path.")
    print(r'Example: C:\Users\puyap\OneDrive\Desktop\CodeToDesignProject\project\data\UmlProjects\usecase\RemoteDesktop\code')
    raw_path = input("PROJECT path: ").strip().strip('"')
    project_code_dir = Path(raw_path).resolve()

    if not validate_project_dir(project_code_dir):
        sys.exit(3)

    # 4) Compute OUT = sibling 'graphs'
    out_dir = compute_out_dir_from_project(project_code_dir)
    print(f"\nResolved paths:\n  PROJECT = {project_code_dir}\n  OUT     = {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Parse & Export (build cpg.bin + cpg_graphson)
    if not run_joern_export(JOERN_IMAGE, project_code_dir, out_dir):
        sys.exit(4)

    # 6) Just create/write the .sc file inside OUT (no Docker run here)
    ok = traversingGraph(
        out_dir=out_dir,
        script_text=DEFAULT_SC_SCRIPT,                 # or: DEFAULT_SC_SCRIPT to auto-write this file now
        script_filename=SC_SCRIPT_NAME,
        overwrite=True,
        verbose=True,
    )
    if not ok:
        sys.exit(5)


    # 7) Run Joern script now (SC → JSON)
    ucd_terms = os.getenv("UCD_TERMS")
    if not run_joern_script(
        image=JOERN_IMAGE,
        out_dir=out_dir,
        script_filename=SC_SCRIPT_NAME,
        out_json_name="usecase_table_metrics_v2.json",
        ucd_terms=ucd_terms,
        verbose=True,
        timeout=3600,
    ):
        sys.exit(6)

    # 7.5) ست کردن کلید از تابع هاردکد
    ensure_openai_api_key(persist=False, key_path=None, interactive=False)

        # 8) Send JSON + prompt to LLM
    prompt_path = Path(__file__).parent / "usecaseprompt.txt"
    llm_obj = send_metrics_to_llm(
        out_dir=out_dir,
        prompt_path=prompt_path,
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        prefer_upload=True,
        verbose=True,
    )

    
    if isinstance(llm_obj, dict):
        extract_from_llm(llm_obj)

        # Use Case Diagram از PlantUML
        if PLANTUML_TEXT:
            func9_generate_plantuml_png(PLANTUML_TEXT, out_dir)

        # UseCaseCodeBindings به testout.txt
        if USECASE_BINDINGS:
            func10_write_bindings_txt(USECASE_BINDINGS, out_file=Path.cwd() / "testout.txt")

    func11_generate_security_report(
    prompt_path=Path(__file__).parent / "securityprompt.txt",
    input_txt_path=Path(__file__).parent / "testout.txt",
    model=os.environ.get("OPENAI_MODEL", "gpt-5"),
    verbose=True,
)

    print("\n✅ All done.")



if __name__ == "__main__":
    main()
